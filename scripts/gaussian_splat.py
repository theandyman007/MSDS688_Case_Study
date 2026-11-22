"""
Stage 2B: Gaussian Splatting branch.

COLMAP poses + sparse points -> seed 3D Gaussians -> optimize with gsplat
(L1 + D-SSIM loss, clone/split/prune densification) -> trained splats .ply

Usage:
python scripts/gaussian_splat.py data/colmap/<scene>                    # 30k steps, half resolution
python scripts/gaussian_splat.py data/colmap/<scene> --resume           # continue after a crash / Ctrl+C
python scripts/gaussian_splat.py data/colmap/<scene> --steps 7000       # quick preview
python scripts/gaussian_splat.py data/colmap/<scene> --data-factor 1    # full resolution (more VRAM)
python scripts/gaussian_splat.py data/colmap/<scene> --test-every 0     # train on every frame
"""

import argparse
import signal
import time
from pathlib import Path
import numpy as np

# PIL MUST be imported before pycolmap. With pycolmap 4.2.1 + Pillow 13, loading pycolmap
# first makes Pillow's PNG (zlib) encoder corrupt the heap ("corrupted size vs. prev_size").
from PIL import Image
import pycolmap

import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

SH_C0 = 0.28209479177387814   # zeroth spherical-harmonic constant: rgb = 0.5 + SH_C0 * sh0




def pose_matrix(image):
    # pycolmap >= 3.12 made cam_from_world a method (rigs/frames); older versions use a property
    cfw = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
    m = np.eye(4)
    m[:3, :4] = cfw.matrix()
    return m                                            # world -> camera, what gsplat expects




def load_scene(colmap_dir, data_factor, test_every):
    workspace = colmap_dir / 'undistorted'              # written by run_sfm.py
    if not (workspace / 'sparse').exists():
        raise RuntimeError(f"{workspace / 'sparse'} not found - run scripts/run_sfm.py first")

    rec = pycolmap.Reconstruction(str(workspace / 'sparse'))
    images, viewmats, Ks, names = [], [], [], []

    for img in sorted(rec.images.values(), key = lambda i: i.name):
        cam = rec.cameras[img.camera_id]
        pil = Image.open(workspace / 'images' / img.name).convert('RGB')
        if data_factor > 1:
            pil = pil.resize((pil.width // data_factor, pil.height // data_factor), Image.LANCZOS)

        # Undistorted cameras are PINHOLE, so K is all we need; rescale it to the loaded image size
        K = np.array(cam.calibration_matrix(), dtype = np.float64)
        K[0, :] *= pil.width / cam.width
        K[1, :] *= pil.height / cam.height

        images.append(torch.from_numpy(np.asarray(pil).copy()))   # uint8 [H, W, 3], kept on CPU
        viewmats.append(pose_matrix(img))
        Ks.append(K)
        names.append(img.name)

    viewmats = np.stack(viewmats)

    # Scene size = how far the cameras spread out; scales the position learning rate and densification
    centers = np.linalg.inv(viewmats)[:, :3, 3]
    scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis = 1).max() * 1.1)

    pts = list(rec.points3D.values())
    xyz = np.array([p.xyz for p in pts], dtype = np.float32)
    rgb = np.array([p.color for p in pts], dtype = np.float32) / 255.0

    idx = np.arange(len(images))
    if test_every > 0:
        test_ids = idx[idx % test_every == 0]
        train_ids = idx[idx % test_every != 0]
    else:
        test_ids, train_ids = idx[:0], idx

    print(f"Loaded {len(images)} images at {images[0].shape[1]}x{images[0].shape[0]}, "
          f"{len(xyz):,} seed points, scene scale {scene_scale:.2f}")
    print(f"{len(train_ids)} train / {len(test_ids)} test views")

    return dict(images = images, viewmats = viewmats, Ks = np.stack(Ks), names = names,
                xyz = xyz, rgb = rgb, scene_scale = scene_scale,
                train_ids = train_ids, test_ids = test_ids)




def init_gaussians(xyz, rgb, scene_scale, sh_degree, device):
    means = torch.from_numpy(xyz).to(device)
    n = len(means)

    # Initial size = average distance to the 3 nearest neighbours (chunked to save VRAM)
    d2 = []
    for chunk in means.split(1024):
        d = torch.cdist(chunk, means)
        d2.append(d.topk(4, largest = False).values[:, 1:].pow(2).mean(1))
    dist = torch.cat(d2).clamp_min(1e-10).sqrt()

    quats = torch.zeros(n, 4, device = device)
    quats[:, 0] = 1.0                                   # identity rotation

    params = torch.nn.ParameterDict({
        'means':     torch.nn.Parameter(means),
        'scales':    torch.nn.Parameter(torch.log(dist)[:, None].repeat(1, 3)),   # stored as log
        'quats':     torch.nn.Parameter(quats),
        'opacities': torch.nn.Parameter(torch.logit(torch.full((n,), 0.1, device = device))),  # logit
        'sh0':       torch.nn.Parameter(((torch.from_numpy(rgb).to(device) - 0.5) / SH_C0)[:, None, :]),
        'shN':       torch.nn.Parameter(torch.zeros(n, (sh_degree + 1) ** 2 - 1, 3, device = device)),
    })

    return params, make_optimizers(params, scene_scale)




def make_optimizers(params, scene_scale):
    # Learning rates from the original 3DGS paper / gsplat's simple_trainer
    lrs = {'means': 1.6e-4 * scene_scale, 'scales': 5e-3, 'quats': 1e-3,
           'opacities': 5e-2, 'sh0': 2.5e-3, 'shN': 2.5e-3 / 20}
    return {name: torch.optim.Adam([{'params': params[name], 'lr': lr, 'name': name}], eps = 1e-15)
            for name, lr in lrs.items()}




def save_checkpoint(path, step, params, optimizers, state, scheduler, config):
    tmp = path.with_suffix('.tmp')
    torch.save({'step': step,
                'config': config,
                'params': {k: v.detach() for k, v in params.items()},
                'optimizers': {k: o.state_dict() for k, o in optimizers.items()},
                'strategy_state': state,
                'scheduler': scheduler.state_dict()}, tmp)
    tmp.replace(path)      # write then rename, so a crash mid-save can't corrupt the last checkpoint
    print(f"Checkpoint saved at step {step + 1} -> {path}")




def load_checkpoint(path, scene_scale, config, device):
    ckpt = torch.load(path, map_location = device, weights_only = False)

    # The LR schedule and densification timing depend on these, so warn if they changed
    for key, value in ckpt.get('config', {}).items():
        if config.get(key) != value:
            print(f"WARNING: checkpoint used {key}={value}, this run uses {key}={config.get(key)}")

    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in ckpt['params'].items()})
    optimizers = make_optimizers(params, scene_scale)
    for k, opt in optimizers.items():
        opt.load_state_dict(ckpt['optimizers'][k])
    return params, optimizers, ckpt




def render(params, viewmat, K, width, height, sh_degree, bg):
    colors = torch.cat([params['sh0'], params['shN']], 1)
    out, _, info = rasterization(
        means = params['means'],
        quats = params['quats'],
        scales = torch.exp(params['scales']),
        opacities = torch.sigmoid(params['opacities']),
        colors = colors,
        viewmats = viewmat[None],
        Ks = K[None],
        width = width,
        height = height,
        sh_degree = sh_degree,
        backgrounds = bg,
        packed = False,
    )
    return out[0], info                                 # [H, W, 3]




def ssim(a, b, window = 11, sigma = 1.5):
    """SSIM between two [H, W, 3] images in 0..1."""
    a, b = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    g = torch.exp(-(torch.arange(window, device = a.device) - window // 2).float() ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).expand(3, 1, window, window).contiguous()
    blur = lambda x: F.conv2d(x, kernel, padding = window // 2, groups = 3)

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return s.mean()




def save_ply(params, path, chunk = 262_144):
    """Standard 3DGS .ply layout (opens in SuperSplat, etc.). Streams in chunks to keep RAM low."""
    n = len(params['means'])
    n_rest = params['shN'].shape[1] * 3
    names = (['x', 'y', 'z', 'nx', 'ny', 'nz']
             + [f'f_dc_{i}' for i in range(3)]
             + [f'f_rest_{i}' for i in range(n_rest)]
             + ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)])
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {n}\n'
              + ''.join(f'property float {name}\n' for name in names)
              + 'end_header\n')

    tmp = Path(path).with_suffix('.ply.tmp')
    with torch.no_grad(), open(tmp, 'wb') as f:
        f.write(header.encode())
        for s in range(0, n, chunk):
            sl = slice(s, s + chunk)
            m = params['means'][sl]
            rows = torch.cat([m, torch.zeros_like(m),
                              params['sh0'][sl, 0, :],
                              params['shN'][sl].transpose(1, 2).reshape(len(m), -1),
                              params['opacities'][sl, None],
                              params['scales'][sl],
                              F.normalize(params['quats'][sl], dim = -1)], 1)
            f.write(rows.float().cpu().numpy().astype('<f4').tobytes())
    tmp.replace(path)
    print(f"Saved {n:,} Gaussians to {path}")




def fmt(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"



def tilt(viewmat, degrees):
    """
    Pitch a world -> camera matrix about the acmera's own x axis.  Negative = look up, positive = look down.
    """
    a = np.radians(degrees)
    R = torch.eye(4, device = viewmat.device)
    R[1, 1], R[1, 2] = np.cos(a), -np.sin(a)
    R[2, 1], R[2, 2] = np.sin(a), np.cos(a)

    return R @ viewmat




def evaluate(params, scene, viewmats, Ks, sh_degree, bg, out_dir, device):
    if len(scene['test_ids']) == 0:
        return

    render_dir = out_dir / 'renders'
    render_dir.mkdir(exist_ok = True)
    tilt_dirs = {'up': -35, 'down': 35}
    for name in tilt_dirs:
        (out_dir / f'renders_{name}').mkdir(exist_ok = True)

    psnrs, ssims = [], []

    with torch.no_grad():
        for i in scene['test_ids']:
            gt = scene['images'][i].to(device).float() / 255
            pred, _ = render(params, viewmats[i], Ks[i], gt.shape[1], gt.shape[0], sh_degree, bg)
            pred = pred.clamp(0, 1)

            psnrs.append((-10 * torch.log10(F.mse_loss(pred, gt))).item())
            ssims.append(ssim(pred, gt).item())

            # Ground truth on the left, render on the right
            side = torch.cat([gt, pred], 1).mul(255).byte().cpu().numpy()
            Image.fromarray(side).save(render_dir / Path(scene['names'][i]).with_suffix('.png').name)
            for name, deg in tilt_dirs.items():
                img, _ = render(params, tilt(viewmats[i], deg), Ks[i], gt.shape[1], gt.shape[0], sh_degree, bg)
                out = img.clamp(0, 1).mul(255).byte().cpu().numpy()
                Image.fromarray(out).save(out_dir / f'renders_{name}' / Path(scene['names'][i]).with_suffix('.png').name)

    msg = (f"Held-out views: {len(psnrs)}  PSNR {np.mean(psnrs):.2f} dB  SSIM {np.mean(ssims):.4f}")
    print(msg)
    (out_dir / 'metrics.txt').write_text(msg + '\n')



def export(colmap_dir, out_dir, data_factor, sh_degree, test_every, white_bg):
    device = 'cuda'
    out_dir = Path(out_dir)
    ckpt_path = out_dir / 'checkpoint.pt'
    if not ckpt_path.exists():
        raise RuntimeError(f"No checkpoint at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False, mmap=True)
    cfg = ckpt.get('config', {})
    data_factor = cfg.get('data_factor', data_factor)       # must match how it was trained
    sh_degree = cfg.get('sh_degree', sh_degree)
    test_every = cfg.get('test_every', test_every)
    params = {k: v.to(device) for k, v in ckpt['params'].items()}
    print(f"Loaded step {ckpt['step'] + 1} checkpoint with {len(params['means']):,} Gaussians")
    del ckpt

    save_ply(params, out_dir / 'splats.ply')

    scene = load_scene(Path(colmap_dir), data_factor, test_every)
    viewmats = torch.tensor(scene['viewmats'], dtype = torch.float32, device = device)
    Ks = torch.tensor(scene['Ks'], dtype = torch.float32, device = device)
    bg = (torch.ones if white_bg else torch.zeros)(1, 3, device = device)
    evaluate(params, scene, viewmats, Ks, sh_degree, bg, out_dir, device)




def train(colmap_dir, out_dir, steps = 30_000, data_factor = 2, sh_degree = 3,
          ssim_lambda = 0.2, grow_grad2d = 0.0002, test_every = 8, white_bg = False, log_every = 500,
          ckpt_every = 5000, resume = False):
    if not torch.cuda.is_available():
        raise RuntimeError('gsplat needs a CUDA GPU (torch.cuda.is_available() is False)')

    device = 'cuda'
    torch.manual_seed(0)
    colmap_dir, out_dir = Path(colmap_dir), Path(out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    ckpt_path = out_dir / 'checkpoint.pt'
    config = {'steps': steps, 'data_factor': data_factor, 'sh_degree': sh_degree, 'test_every': test_every, 'grow_grad2d': grow_grad2d}

    scene = load_scene(colmap_dir, data_factor, test_every)
    viewmats = torch.tensor(scene['viewmats'], dtype = torch.float32, device = device)
    Ks = torch.tensor(scene['Ks'], dtype = torch.float32, device = device)
    bg = (torch.ones if white_bg else torch.zeros)(1, 3, device = device)

    # Clone/split Gaussians with large screen-space gradients, prune transparent/huge ones,
    # and reset opacity every 3000 steps; densification stops halfway through training
    strategy = DefaultStrategy(grow_grad2d = grow_grad2d, refine_stop_iter = min(15_000, steps // 2), verbose = True)

    ckpt = None
    if resume and ckpt_path.exists():
        params, optimizers, ckpt = load_checkpoint(ckpt_path, scene['scene_scale'], config, device)
        state = ckpt['strategy_state']
        start_step = ckpt['step'] + 1
        print(f"Resuming from step {start_step} with {len(params['means']):,} Gaussians")
    else:
        if resume:
            print(f"No checkpoint at {ckpt_path} - starting from scratch")
        params, optimizers = init_gaussians(scene['xyz'], scene['rgb'], scene['scene_scale'], sh_degree, device)
        state = strategy.initialize_state(scene_scale = scene['scene_scale'])
        start_step = 0

    strategy.check_sanity(params, optimizers)

    # Position learning rate decays 100x over training
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers['means'], gamma = 0.01 ** (1 / steps))
    if ckpt is not None:
        scheduler.load_state_dict(ckpt['scheduler'])
        del ckpt

    # First Ctrl+C: finish the current step, save a checkpoint, then stop cleanly.
    # Second Ctrl+C: quit immediately.
    stop = {'requested': False}
    def on_sigint(signum, frame):
        if stop['requested']:
            raise KeyboardInterrupt
        stop['requested'] = True
        print("\nCtrl+C - finishing this step and saving (press again to quit immediately)")
    signal.signal(signal.SIGINT, on_sigint)

    train_ids = scene['train_ids']
    t0 = time.time()
    step = start_step - 1

    try:
        for step in range(start_step, steps):
            i = int(train_ids[np.random.randint(len(train_ids))])
            gt = scene['images'][i].to(device, non_blocking = True).float() / 255

            # Add one spherical-harmonic band every 1000 steps (view-dependent colour)
            sh_now = min(step // 1000, sh_degree)
            pred, info = render(params, viewmats[i], Ks[i], gt.shape[1], gt.shape[0], sh_now, bg)

            strategy.step_pre_backward(params, optimizers, state, step, info)

            l1 = (pred - gt).abs().mean()
            loss = (1 - ssim_lambda) * l1 + ssim_lambda * (1 - ssim(pred, gt))
            loss.backward()

            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none = True)
            scheduler.step()

            strategy.step_post_backward(params, optimizers, state, step, info, packed = False)

            if (step + 1) % log_every == 0 or step == start_step:
                elapsed = time.time() - t0
                eta = elapsed / (step + 1 - start_step) * (steps - step - 1)
                print(f"step {step + 1:>6}/{steps}  loss {loss.item():.4f}  "
                      f"gaussians {len(params['means']):>9,}  "
                      f"vram {torch.cuda.max_memory_allocated() / 1e9:.1f} GB  "
                      f"elapsed {fmt(elapsed)}  eta {fmt(eta)}")

            if ckpt_every > 0 and (step + 1) % ckpt_every == 0:
                save_checkpoint(ckpt_path, step, params, optimizers, state, scheduler, config)

            if stop['requested']:
                save_checkpoint(ckpt_path, step, params, optimizers, state, scheduler, config)
                print(f"Stopped at step {step + 1}. Rerun with --resume to continue.")
                break

    except Exception:
        # Crash (out of memory, CUDA error, ...): the last periodic checkpoint is still valid
        print(f"\nTraining failed at step {step + 1}.")
        if ckpt_path.exists():
            print(f"Last good checkpoint: {ckpt_path} - rerun with --resume to continue from it.")
        raise

    print(f"Training session took {fmt(time.time() - t0)}")
    save_ply(params, out_dir / 'splats.ply')

    if not stop['requested']:
        evaluate(params, scene, viewmats, Ks, sh_degree, bg, out_dir, device)

    return params




if __name__ == "__main__":
    p = argparse.ArgumentParser(description = 'Stage 2B: 3D Gaussian Splatting with gsplat')
    p.add_argument('colmap_dir', help = 'Stage 1 output folder (data/colmap/<scene>)')
    p.add_argument('--out', default = None, help = 'default: data/splats/<scene name>')
    p.add_argument('--steps', type = int, default = 30_000)
    p.add_argument('--data-factor', type = int, default = 2,
                   help = 'downscale images by this factor (1 = full size, more VRAM and slower)')
    p.add_argument('--sh-degree', type = int, default = 3)
    p.add_argument('--ssim-lambda', type = float, default = 0.2)
    p.add_argument('--grow-grad2d', type = float, default = 0.0002,
                   help = 'densification threshold; higher = fewer Gaussians, less VRAM')
    p.add_argument('--test-every', type = int, default = 8,
                   help = 'hold out every Nth frame for PSNR/SSIM (0 = train on all)')
    p.add_argument('--white-bg', action = 'store_true', help = 'white background (helps with open sky)')
    p.add_argument('--ckpt-every', type = int, default = 5000,
                   help = 'save a checkpoint every N steps (0 = off)')
    p.add_argument('--resume', action = 'store_true', help = 'continue from <out>/checkpoint.pt')
    p.add_argument('--export-only', action = 'store_true',
                   help = 'no training: write splats.ply + metrics from <out>/checkpoint.pt')
    args = p.parse_args()

    out = args.out or f"data/splats/{Path(args.colmap_dir).name}"
    if args.export_only:
        export(args.colmap_dir, out, args.data_factor, args.sh_degree, args.test_every, args.white_bg)
    else:
        train(args.colmap_dir, out, steps = args.steps, data_factor = args.data_factor,
              sh_degree = args.sh_degree, ssim_lambda = args.ssim_lambda,
              grow_grad2d = args.grow_grad2d, test_every = args.test_every,
              white_bg = args.white_bg, ckpt_every = args.ckpt_every, resume = args.resume)