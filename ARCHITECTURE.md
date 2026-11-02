# Physics-Informed Residual Hand Tracker — Architecture Specification

## 1. Coordinate Frames and Conventions
* **Camera Coordinate System ($\mathcal{F}_C$)**:
  * Standard OpenCV/Pinhole convention: $+X$ points right, $+Y$ points down, $+Z$ points forward along the optical axis into the scene.
  * Origin $(0, 0, 0)$ is situated at the camera optical center.
* **World Coordinate System ($\mathcal{F}_W$)**:
  * Right-handed Cartesian reference frame. By default, aligned with $\mathcal{F}_C$ unless an explicit extrinsic transform $(R_{WC}, t_{WC})$ is supplied.
  * Distance units: meters ($m$). Time units: seconds ($s$).
* **Local Hand Root System ($\mathcal{F}_H$)**:
  * Origin $(0, 0, 0)$ is anchored at the Wrist (Landmark 0).
  * $+Y$ extends along the palm towards the middle metacarpophalangeal (MCP) joint.
  * $+X$ extends laterally across the palm towards the thumb base (radial direction).
  * $+Z$ points outward normally from the dorsal surface of the hand.
* **Orientation Representation**:
  * Hamilton Unit Quaternion $q = [w, x, y, z] \in \mathbb{H}_1$, where $w^2 + x^2 + y^2 + z^2 = 1$.
  * Rotation operation on vector $v$: $v' = q \otimes [0, v] \otimes q^*$.
* **Image Coordinate Frame ($\mathcal{F}_I$)**:
  * Pixel coordinates $(u, v)$ with origin $(0, 0)$ at the top-left corner.
  * $+u$ extends along image width (columns); $+v$ extends along image height (rows).

## 2. Anatomical Topology & Degrees of Freedom
The hand is modeled as an articulated kinematic tree comprising 21 standard anatomical joints:
* **Root (Joint 0)**: Wrist (6-DoF global pose: $T \in \mathbb{R}^3, R \in SO(3)$).
* **Thumb Chain**:
  * Joint 1: CMC (Carpometacarpal) — 2 DoF (abduction/adduction, flexion/extension).
  * Joint 2: MCP (Metacarpophalangeal) — 2 DoF (abduction/adduction, flexion/extension).
  * Joint 3: IP (Interphalangeal) — 1 DoF (flexion/extension).
  * Joint 4: TIP (Thumb Tip) — 0 DoF leaf node.
* **Fingers (Index, Middle, Ring, Pinky)**:
  * MCP (Joints 5, 9, 13, 17) — 2 DoF each (abduction/adduction, flexion/extension).
  * PIP (Joints 6, 10, 14, 18) — 1 DoF each (flexion/extension).
  * DIP (Joints 7, 11, 15, 19) — 1 DoF each (flexion/extension).
  * TIP (Joints 8, 12, 16, 20) — 0 DoF leaf nodes.
* **Total Internal Articulation**: $2 + 2 + 1 + 4 \times (2 + 1 + 1) = 21$ DoF joint space ($q \in \mathbb{R}^{21}$).
* **Total Dynamic State Dimension**:
  $$\dim(X) = 3 + 4 + 21 + 3 + 3 + 21 = 55 \quad \text{with } q_{\text{rot}} \in \mathbb{H}_1$$

## 3. Kinematic and Geometric Guarantees
1. **Bone Length Invariance**: Segment distances $\|p_{\text{child}} - p_{\text{parent}}\|_2 \equiv L_{\text{bone}}$ are invariant under all pose configurations.
2. **Singularity-Free $SO(3)$ Dynamics**: Global angular integration uses the matrix exponential map on $SO(3)$ via unit quaternions:
   $$q_t^- = \exp\left(\frac{1}{2}\omega_{t-1}\Delta t\right) \otimes q_{t-1}$$
3. **Hard Anatomical Projections**: Joint limits enforce $[q_{\min}, q_{\max}]$ bounds, and velocity clamping ensures $\|\dot{q}\|_\infty \le \dot{q}_{\max}$.
4. **Analytical Glove Representation**: Bone segments are enclosed by analytical capsule primitives with physiologically scaled radii ($r_{\text{proximal}} \to r_{\text{distal}}$).