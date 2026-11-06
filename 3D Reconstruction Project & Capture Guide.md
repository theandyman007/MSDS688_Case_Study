# End-to-End 3D Scene Reconstruction: Project Blueprint & Capture Guide
**Photogrammetry vs. 3D Gaussian Splatting (3DGS) in Python**

---

## 1. Overarching Project Outlook & Core Concepts

### The Core Premise
Both **Photogrammetry** and **3D Gaussian Splatting (3DGS)** transform a set of 150–300 overlapping 2D photographs (extracted from a continuous smartphone video) into a 3D representation of a real-world space. However, they use fundamentally different mathematics and serve opposite practical goals:

*   **Photogrammetry (The Sculptor — Explicit Surface Geometry):**
    *   **Math:** Identifies matching pixel features across overlapping images and uses geometric triangulation to compute exact $(X, Y, Z)$ surface coordinates. It connects these points into a wireframe of flat triangles (**polygon mesh**) and wraps a 2D image (**UV texture**) over the surface.
    *   **Strengths:** Excels at hard, matte, textured surfaces (bricks, statues, rocks, furniture). Produces solid, measurable geometry that supports physics collisions, CAD measurements, and 3D printing.
    *   **Weaknesses:** Fails on shiny, transparent, reflective, or paper-thin structures (glass, mirrors, car paint, bike spokes, leaves), turning them into jagged, melted blobs. Ignores or mangles distant backgrounds.
*   **3D Gaussian Splatting (The Impressionist Painter — Volumetric Radiance):**
    *   **Math:** Populates 3D space with millions of semi-transparent, fuzzy 3D ellipsoids ("splats") defined by 3D Gaussian distributions:
        $$G(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$
        Using differentiable rasterization and gradient descent (`loss.backward()`), a neural/optimization loop stretches, rotates, adjusts opacity ($\alpha$), and tunes view-dependent color (**spherical harmonics**) for each splat until rendering the cloud from any camera angle matches your original photographs.
    *   **Strengths:** Effortlessly captures reflections, glass, fine foliage, thin wires, ambient lighting, and entire background environments at $60+$ FPS.
    *   **Weaknesses:** It is a "cloud of light blobs," not a solid surface. Up close, flat walls can look slightly fuzzy, and game characters will fall right through the floor unless paired with a collision mesh.

---

## 2. Project Goals & Architecture

### Primary Objective
Build a modular, automated **Python pipeline** (without relying on commercial black-box GUI apps) that takes a raw smartphone video (`.mp4` or `.mov`), extracts the sharpest frames, solves the 3D camera trajectory, and outputs **both** a Photogrammetry surface mesh and a 3D Gaussian Splat scene from the exact same dataset.

### The 4-Stage Python Tech Stack

| Stage | Module / Library | Input $\rightarrow$ Output | What the Code Does |
| :--- | :--- | :--- | :--- |
| **Stage 0: Sharp Frame Slicing** | `opencv-python` (`cv2`) | `.mp4` Video $\rightarrow$ 150–300 `.jpg` frames | Divides the video into $N$ evenly spaced time windows and selects the frame with the highest **Variance of the Laplacian** ($\text{Var}(\nabla^2 I)$) in each window to eliminate motion blur. |
| **Stage 1: Structure-from-Motion (SfM)** | `pycolmap` | `.jpg` folder $\rightarrow$ Camera Poses $(R, \mathbf{t})$ + Sparse Point Cloud | Extracts SIFT keypoints, matches sequential overlapping frames, and runs incremental bundle adjustment to solve the exact 3D position and angle of the phone for every frame. |
| **Stage 2A: Photogrammetry Branch** | `open3d` + `pycolmap` | Sparse/Dense Points $\rightarrow$ `.obj` / `.ply` Mesh | Cleans statistical outlier points, estimates surface normals, and executes **Screened Poisson Surface Reconstruction** to generate a watertight polygon mesh. |
| **Stage 2B: Gaussian Splatting Branch** | `torch` + `gsplat` | COLMAP Points + Images $\rightarrow$ Trained `.ply` Splats | Seeds 3D Gaussians at the COLMAP points and runs a PyTorch optimization loop minimizing photometric loss: $$\mathcal{L} = (1 - \lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$$ while cloning and splitting Gaussians in high-gradient regions. |

### Portfolio Extensions & Experiments
1.  **The "Torture Test" Benchmark:** Run both branches on a scene containing both matte geometry and reflective/thin objects (or benchmark datasets like **Mip-NeRF 360** and **Tanks & Temples**). Quantify novel-view synthesis quality on held-out test views (every 8th frame) using **PSNR**, **SSIM**, and **LPIPS**.
2.  **Frame-Count Ablation Study:** Evaluate how reconstruction quality degrades as you downsample input frames ($300 \rightarrow 150 \rightarrow 75 \rightarrow 30$ frames).
3.  **Hybrid Scene Integration:** Combine the invisible Photogrammetry `.obj` mesh (for physics collisions) with the visible 3DGS `.ply` splat (for photorealistic rendering).

---

## 3. Field Guide: Recording Your First Video

### The Golden Rule: Translation Over Rotation (Never Spin in Place!)
To calculate 3D depth, `pycolmap` requires **parallax**—closer objects shifting across the screen faster than distant background objects.
*   **NEVER stand in one spot and pivot/rotate your body.** If your camera stays at a fixed $(X, Y, Z)$ coordinate, the baseline distance between frames is zero ($b \approx 0$), resulting in zero depth triangulation and a fatal COLMAP failure.
*   **ALWAYS walk physically** so your feet move the camera through 3D space.

---

### Camera Pre-Flight Checklist (Before Pressing Record)
1.  **Lens Selection:** Use your phone's **1x (Main Wide)** or **0.5x (Ultra-Wide)** lens at **4K 60fps**. Never zoom in during the video.
2.  **Disable Stabilization:** Turn **OFF** "Action Mode," "Enhanced Stabilization," or digital EIS in your camera settings. Digital stabilization dynamically crops and warps the frame edges, breaking `pycolmap`'s pinhole camera model.
3.  **Lock Exposure & Focus (AE/AF Lock):** Point your phone at a medium-lit area of the scene and **tap-and-hold the screen until `AE/AF Lock` appears**. Brightness and focal length must remain frozen for the entire walk.
4.  **Move Slowly:** Walk at roughly **one slow step per second**, keeping the phone steady with both hands. Motion blur is the #1 enemy of feature matching.

---

### The "Three Donuts" Walking Pattern (Continuous 90–120s Video)
Record all three loops in **one continuous video take** so `pycolmap.match_sequential()` can track your trajectory seamlessly from start to finish:

1.  **Loop 1 — Eye-Level Pass ($0^\circ$ pitch):** Walk a full $360^\circ$ circle at chest/eye height pointing straight ahead at your subject or across the room.
2.  **Loop 2 — High-Angle Overhead Pass ($-45^\circ$ downward pitch):** Without stopping the recording, raise the phone above your head, angle it downward at roughly $45^\circ$, and walk a second $360^\circ$ loop to capture tabletops, seats, and floor surfaces.
3.  **Loop 3 — Low-Angle Waist Pass ($+15^\circ$ upward pitch):** Lower the phone to waist/knee height, angle it slightly upward, and walk a third $360^\circ$ loop to capture underneath desk edges, chair legs, and upper walls.

---

### Indoor vs. Outdoor Capture Strategies

| Environment | Walking Trajectory | Environment-Specific Rules & Traps to Avoid |
| :--- | :--- | :--- |
| **Indoor: Full Room** | **Outward Perimeter Loop:** Stand near the wall and walk around the outer perimeter of the room while pointing the camera **inward across the center** toward the opposite wall. | • **Turn on every light** in the room so your phone uses a fast shutter speed.<br>• **Avoid the "Blank Wall Trap":** Never let a plain white wall, ceiling, or blank door fill $100\%$ of the screen. Always keep a textured edge (poster, desk, bookshelf, rug) visible in the frame.<br>• Turn off ceiling fans and keep pets out of the room. |
| **Indoor/Outdoor: Single Centerpiece** *(Desk, Statue, Gazebo, Bench)* | **Inward Orbit:** Step back 5–10 feet from the centerpiece and walk 3 full $360^\circ$ circles around the outside of it, pointing **inward** at the object the entire time. | • Captures $360^\circ$ solid geometry of the centerpiece while 3DGS naturally reconstructs the surrounding park or room in the background.<br>• Watch your own shadow so it doesn't sweep across the object as you orbit. |
| **Outdoor: Park Clearing or Coffee Shop Patio** | **Boundary Loop or $180^\circ$ Crab-Walk Arc:** Walk around the outer boundary pointing inward across the patio, or walk in multiple sideways arcs in front of a storefront. | • **Best Weather:** Shoot on an **overcast/cloudy day** or in even shade. Direct midday sun creates harsh shadows and lens flares.<br>• **Wind Check:** Avoid windy days where swaying branches dominate the frame; always keep solid ground anchors (pavement, bricks, trunks, patio tables) in the lower half of the shot.<br>• **Strangers Walking By:** If someone walks right in front of your lens, pause your feet for 3 seconds until they pass, then resume walking from that exact spot (your Python script can drop those obstructed frames).<br>• **The Infinite Sky Fix:** Avoid tilting the camera steeply into empty sky. In `gsplat`, pass a white/sky background tensor (`backgrounds=torch.ones(1, 3, device=device)`) so the trainer doesn't spawn giant floating white splats to fake the sky. |

---

## 4. Reference Links & Public Datasets (For Testing)
*   **Mip-NeRF 360 Benchmark Dataset:** [https://nerfbaselines.github.io/m-mipnerf360](https://nerfbaselines.github.io/m-mipnerf360)
*   **Tanks and Temples Benchmark Dataset:** [https://www.tanksandtemples.org/download/](https://www.tanksandtemples.org/download/)
*   **PyCOLMAP Documentation:** [https://colmap.github.io/pycolmap/pycolmap.html](https://colmap.github.io/pycolmap/pycolmap.html)
*   **gsplat Documentation:** [https://docs.gsplat.studio/main/](https://docs.gsplat.studio/main/)
*   **New York Times R&D — Field Guide to Gaussian Splatting:** [https://rd.nytimes.com/projects/gaussian-splatting-guide/](https://rd.nytimes.com/projects/gaussian-splatting-guide/)
*   **Garage Farm — Gaussian Splatting vs. Photogrammetry Overview:** [https://garagefarm.net/blog/gaussian-splatting-vs-photogrammetry-which-3d-reconstruction-method-should-you-use](https://garagefarm.net/blog/gaussian-splatting-vs-photogrammetry-which-3d-reconstruction-method-should-you-use)
*   **Splatware — Photogrammetry vs. Gaussian Splatting Guide:** [https://splatware.com/learn/photogrammetry-vs-gaussian-splatting](https://splatware.com/learn/photogrammetry-vs-gaussian-splatting)
*   **KIRI Engine — 3DGS vs. Photogrammetry vs. LiDAR Decision Guide:** [https://www.kiriengine.app/blog/3DGSvsPhotogrammetryvsLiDAR](https://www.kiriengine.app/blog/3DGSvsPhotogrammetryvsLiDAR)