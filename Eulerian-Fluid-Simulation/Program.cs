using Raylib_cs;
using System;
using System.Numerics;
using System.Diagnostics;

public class Program
{
    // ---------------- Grid & render ----------------
    const int NX = 200;
    const int NY = 200;
    const float CELL_SIZE = 3.0f; // pixels per cell
    const float DT = 0.2f;

    // ---------------- Physics params ----------------
    const float RHO = 1.0f;
    const float GRAVITY_X = 0.0f;
    const float GRAVITY_Y = -9.0f;
    const float VISCOSITY = 0.0005f;
    const int DIFFUSION_ITERS = 10;

    // CG options (Stage 2)
    const int CG_MAX_ITERS = 60;
    const float CG_REL_TOL = 1e-4f; // ||r||/||b||

    // ---------------- Flattened fields ----------------
    // Scalars (cell centers)
    static float[] pressure = new float[NX * NY];
    static float[] density = new float[NX * NY];
    static float[] pressureTmp = new float[NX * NY];
    static float[] rhs = new float[NX * NY];
    static float[] divergence = new float[NX * NY];
    static float[] divergencePost = new float[NX * NY];

    // Staggered velocities
    static readonly int NU = (NX + 1) * NY;   // u: (Nx+1, Ny)
    static readonly int NV = NX * (NY + 1);   // v: (Nx, Ny+1)
    static float[] u = new float[NU];
    static float[] v = new float[NV];
    static float[] uTmp = new float[NU];
    static float[] vTmp = new float[NV];
    static float[] u0 = new float[NU]; // diffusion RHS
    static float[] v0 = new float[NV];

    // CG work arrays (Stage 2)
    static float[] cg_r = new float[NX * NY];
    static float[] cg_z = new float[NX * NY];
    static float[] cg_d = new float[NX * NY];
    static float[] cg_Ap = new float[NX * NY];
    static float[] cg_diagInv = new float[NX * NY]; // 1 / diag(L)

    // ---------------- Texture for density (Stage 1) ----------------
    static Texture2D densityTex;
    static byte[] pixels = new byte[NX * NY * 4];

    // ---------------- Input & diagnostics ----------------
    static Vector2 prevMouse;
    const float MOUSE_FORCE_SCALE = 15.0f;
    static Diagnostics diag = new Diagnostics();
    static bool paused = false;
    static bool singleStep = false;
    static bool showVel = true;

    // ---------------- Helpers: indexing ----------------
    static int Idx(int i, int j) => i + j * NX;                 // scalar
    static int IU(int i, int j) => i + j * (NX + 1);            // u faces: 0..NX, 0..NY-1
    static int IV(int i, int j) => i + j * NX;                  // v faces: 0..NX-1, 0..NY

    // ---------------- Solid (Level A) fields ----------------
    static Vector2 circleCenter = new Vector2(NX * 0.5f, NY * 0.5f);
    static float circleRadius = MathF.Min(NX, NY) * 0.12f; // default
    static Vector2 circleVel = Vector2.Zero; // object velocity (static by default)
    static byte[] solidCell = new byte[NX * NY];
    static byte[] solidU = new byte[NU];
    static byte[] solidV = new byte[NV];
    static float[] phiCell = new float[NX * NY];
    static float[] phiU = new float[NU];
    static float[] phiV = new float[NV];
    static int anchorIndex = 0;

    public static void Main()
    {
        int screenW = (int)(NX * CELL_SIZE);
        int screenH = (int)(NY * CELL_SIZE);

        Raylib.InitWindow(screenW, screenH, "2D Eulerian Fluid (MAC) — collisions (Level A) + CG + diagnostics");
        Raylib.SetTargetFPS(120);
        prevMouse = Raylib.GetMousePosition();

        // Stage 1: build density texture
        Image img = Raylib.GenImageColor(NX, NY, Color.Black);
        densityTex = Raylib.LoadTextureFromImage(img);
        Raylib.UnloadImage(img);

        // initial precompute (will be updated per-frame)
        PrecomputeCGDiagonal();

        while (!Raylib.WindowShouldClose())
        {
            // Controls
            if (Raylib.IsKeyPressed(KeyboardKey.D)) diag.toggleShow();
            if (Raylib.IsKeyPressed(KeyboardKey.P)) paused = !paused;
            if (Raylib.IsKeyPressed(KeyboardKey.N)) singleStep = true;
            if (Raylib.IsKeyPressed(KeyboardKey.C)) diag.ClearHistory();
            if (Raylib.IsKeyPressed(KeyboardKey.V)) showVel = !showVel;

            if (!paused || singleStep)
            {
                Update();
                singleStep = false;
            }
            Draw();
        }

        Raylib.UnloadTexture(densityTex);
        Raylib.CloseWindow();
    }

    // ============================================================
    //                            UPDATE
    // ============================================================
    static void Update()
    {
        var swFrame = Stopwatch.StartNew();
        float dt = DT;

        // --- Paint density (LMB) ---
        if (Raylib.IsMouseButtonDown(MouseButton.Left))
        {
            Vector2 mouse = Raylib.GetMousePosition();
            int gx = (int)(mouse.X / CELL_SIZE);
            int gy = (int)(mouse.Y / CELL_SIZE);
            int radius = 6;

            for (int y = gy - radius; y <= gy + radius; y++)
                for (int x = gx - radius; x <= gx + radius; x++)
                {
                    if (x < 0 || x >= NX || y < 0 || y >= NY) continue;
                    int dx = x - gx, dy = y - gy;
                    float distSq = dx * dx + dy * dy;
                    if (distSq <= radius * radius)
                    {
                        float falloff = 1.0f - MathF.Sqrt(distSq) / (radius + 1e-6f);
                        int id = Idx(x, y);
                        density[id] = MathF.Min(1.0f, density[id] + 1.0f * falloff);
                    }
                }
        }

        // --- Inject velocity (RMB drag) ---
        Vector2 mouseNow = Raylib.GetMousePosition();
        if (Raylib.IsMouseButtonDown(MouseButton.Right))
        {
            Vector2 dm = (mouseNow - prevMouse) / CELL_SIZE;
            Vector2 gridPos = new Vector2(mouseNow.X / CELL_SIZE, mouseNow.Y / CELL_SIZE);
            InjectVelocity(gridPos, dm * MOUSE_FORCE_SCALE);
        }
        prevMouse = mouseNow;

        // --- Circle controls ---
        // move circle with middle mouse
        if (Raylib.IsMouseButtonDown(MouseButton.Middle))
        {
            Vector2 m = Raylib.GetMousePosition();
            circleCenter = new Vector2(m.X / CELL_SIZE, m.Y / CELL_SIZE);
        }
        // radius adjust: Z decrease, X increase
        if (Raylib.IsKeyDown(KeyboardKey.Z)) circleRadius = MathF.Max(1.0f, circleRadius - 0.2f);
        if (Raylib.IsKeyDown(KeyboardKey.X)) circleRadius = MathF.Min(MathF.Min(NX, NY) * 0.45f, circleRadius + 0.2f);

        // --- Build level set & masks for the circle (Level A) ---
        BuildLevelSetCircle(circleCenter, circleRadius);
        UpdateCGDiagonalFromMask();   // update diag for CG preconditioner based on masks
        ApplySolidFaceBCs(circleVel); // enforce object face normal speeds before advection/diffusion

        // --- Advection ---
        var swAdv = Stopwatch.StartNew();
        AdvectDensity(dt);
        AdvectVelocity(dt);
        swAdv.Stop();

        // After advection, remove any density inside the solid
        ZeroDensityInSolidCells();

        // --- Forces ---
        var swForces = Stopwatch.StartNew();
        ApplyForces(dt);
        swForces.Stop();

        // --- Diffusion (implicit Jacobi on u,v) ---
        var swDiff = Stopwatch.StartNew();
        if (VISCOSITY > 0.0f) DiffuseVelocities(dt);
        swDiff.Stop();

        // --- Boundaries pre-projection ---
        var swB1 = Stopwatch.StartNew();
        EnforceBoundaries();
        ApplySolidFaceBCs(circleVel); // re-apply (defensive)
        swB1.Stop();

        // --- Divergence & RHS ---
        var swDiv = Stopwatch.StartNew();
        ComputeDivergenceAndRHS_WithSolid(dt);
        swDiv.Stop();

        // --- Pressure solve (Stage 2: CG) ---
        var swPress = Stopwatch.StartNew();
        (int cgIters, float cgRes) = SolvePressureCG(CG_MAX_ITERS, CG_REL_TOL);
        swPress.Stop();

        // --- Projection ---
        var swProj = Stopwatch.StartNew();
        ProjectVelocities_WithSolid(dt);
        swProj.Stop();

        // --- Final boundaries ---
        var swB2 = Stopwatch.StartNew();
        EnforceBoundaries();
        ApplySolidFaceBCs(circleVel); // final safeguard
        swB2.Stop();

        swFrame.Stop();

        // Post-projection divergence for diagnostics
        ComputeDivergencePost();

        // NaN/Inf guard
        if (ScanForNaNInf())
        {
            paused = true;
            Console.WriteLine("NaN/Inf detected — simulation paused.");
        }

        // Diagnostics
        diag.GatherAll(
            density, u, v,
            divergence, divergencePost,
            pressure, cgRes,
            dt,
            swAdv.ElapsedMilliseconds,
            swForces.ElapsedMilliseconds,
            swDiff.ElapsedMilliseconds,
            swDiv.ElapsedMilliseconds,
            swPress.ElapsedMilliseconds,
            swProj.ElapsedMilliseconds,
            swB1.ElapsedMilliseconds + swB2.ElapsedMilliseconds,
            swFrame.ElapsedMilliseconds,
            cgIters
        );
    }

    // ============================================================
    //                         ADVECTION
    // ============================================================
    static void AdvectDensity(float dt)
    {
        for (int j = 0; j < NY; j++)
            for (int i = 0; i < NX; i++)
            {
                Vector2 x = new Vector2(i + 0.5f, j + 0.5f);
                Vector2 v1 = SampleVelocity(x);
                Vector2 xmid = x - 0.5f * dt * v1;
                Vector2 v2 = SampleVelocity(ClampToDomainScalar(xmid));
                Vector2 xprev = ClampToDomainScalar(x - dt * v2);
                density[Idx(i, j)] = SampleScalar(density, xprev);
            }
    }

    static void AdvectVelocity(float dt)
    {
        // u faces
        for (int j = 0; j < NY; j++)
            for (int i = 0; i <= NX; i++)
            {
                Vector2 x = new Vector2(i, j + 0.5f);
                Vector2 v1 = SampleVelocity(x);
                Vector2 xmid = x - 0.5f * dt * v1;
                Vector2 v2 = SampleVelocity(ClampToDomainU(xmid));
                Vector2 xprev = ClampToDomainU(x - dt * v2);
                uTmp[IU(i, j)] = SampleU(u, xprev);
            }

        // v faces
        for (int j = 0; j <= NY; j++)
            for (int i = 0; i < NX; i++)
            {
                Vector2 x = new Vector2(i + 0.5f, j);
                Vector2 v1 = SampleVelocity(x);
                Vector2 xmid = x - 0.5f * dt * v1;
                Vector2 v2 = SampleVelocity(ClampToDomainV(xmid));
                Vector2 xprev = ClampToDomainV(x - dt * v2);
                vTmp[IV(i, j)] = SampleV(v, xprev);
            }

        // swap
        (u, uTmp) = (uTmp, u);
        (v, vTmp) = (vTmp, v);
    }

    // ============================================================
    //                 SAMPLING (bilinear, flattened)
    // ============================================================
    static Vector2 SampleVelocity(Vector2 pos)
    {
        return new Vector2(SampleU(u, pos), SampleV(v, pos));
    }

    static float SampleScalar(float[] A, Vector2 pos)
    {
        // scalar index space
        float xs = pos.X - 0.5f, ys = pos.Y - 0.5f;
        return Bilinear(A, xs, ys, NX, NY);
    }

    static float SampleU(float[] U, Vector2 pos)
    {
        // (i, j+0.5) with size (NX+1, NY)
        float xu = pos.X;
        float yu = pos.Y - 0.5f;
        return Bilinear(U, xu, yu, NX + 1, NY);
    }

    static float SampleV(float[] V, Vector2 pos)
    {
        // (i+0.5, j) with size (NX, NY+1)
        float xv = pos.X - 0.5f;
        float yv = pos.Y;
        return Bilinear(V, xv, yv, NX, NY + 1);
    }

    static float Bilinear(float[] A, float x, float y, int SX, int SY)
    {
        const float EPS = 1e-4f;
        float xc = MathF.Max(0.0f, MathF.Min(x, SX - 1 - EPS));
        float yc = MathF.Max(0.0f, MathF.Min(y, SY - 1 - EPS));

        int i0 = (int)MathF.Floor(xc);
        int j0 = (int)MathF.Floor(yc);
        int i1 = Math.Min(i0 + 1, SX - 1);
        int j1 = Math.Min(j0 + 1, SY - 1);

        float sx = xc - i0;
        float sy = yc - j0;

        int idx00 = i0 + j0 * SX;
        int idx10 = i1 + j0 * SX;
        int idx01 = i0 + j1 * SX;
        int idx11 = i1 + j1 * SX;

        float a00 = A[idx00];
        float a10 = A[idx10];
        float a01 = A[idx01];
        float a11 = A[idx11];

        float v0 = a00 * (1 - sx) + a10 * sx;
        float v1 = a01 * (1 - sx) + a11 * sx;
        return v0 * (1 - sy) + v1 * sy;
    }

    // Domains
    static Vector2 ClampToDomainScalar(Vector2 p) =>
        new Vector2(MathF.Max(0.5f, MathF.Min(p.X, NX - 0.5f)),
                    MathF.Max(0.5f, MathF.Min(p.Y, NY - 0.5f)));
    static Vector2 ClampToDomainU(Vector2 p) =>
        new Vector2(MathF.Max(0.0f, MathF.Min(p.X, NX)),
                    MathF.Max(0.5f, MathF.Min(p.Y, NY - 0.5f)));
    static Vector2 ClampToDomainV(Vector2 p) =>
        new Vector2(MathF.Max(0.5f, MathF.Min(p.X, NX - 0.5f)),
                    MathF.Max(0.0f, MathF.Min(p.Y, NY)));

    // ============================================================
    //                 FORCES / DIFFUSION / PROJECTION
    // ============================================================
    static void ApplyForces(float dt)
    {
        // u: add gx * dt
        for (int j = 0; j < NY; j++)
            for (int i = 0; i <= NX; i++)
                u[IU(i, j)] += dt * GRAVITY_X;

        // v: add gy * dt
        for (int j = 0; j <= NY; j++)
            for (int i = 0; i < NX; i++)
                v[IV(i, j)] += dt * GRAVITY_Y;
    }

    static void EnforceBoundaries()
    {
        // No-flow box
        for (int j = 0; j < NY; j++)
        {
            u[IU(0, j)] = 0.0f;
            u[IU(NX, j)] = 0.0f;
        }
        for (int i = 0; i < NX; i++)
        {
            v[IV(i, 0)] = 0.0f;
            v[IV(i, NY)] = 0.0f;
        }
    }

    static void DiffuseVelocities(float dt)
    {
        float alpha = VISCOSITY * dt;
        if (alpha <= 0.0f) return;

        Array.Copy(u, u0, NU);
        Array.Copy(v, v0, NV);

        for (int iter = 0; iter < DIFFUSION_ITERS; iter++)
        {
            // u
            for (int j = 0; j < NY; j++)
                for (int i = 0; i <= NX; i++)
                {
                    int id = IU(i, j);
                    // if face is solid, keep fixed (Dirichlet)
                    if (solidU[id] != 0) { uTmp[id] = u0[id]; continue; }

                    float sumN = 0f; int N = 0;
                    // left
                    if (i > 0)
                    {
                        int nid = IU(i - 1, j);
                        sumN += u[nid];
                        if (solidU[nid] == 0) N++;
                    }
                    // right
                    if (i < NX)
                    {
                        int nid = IU(i + 1, j);
                        sumN += u[nid];
                        if (solidU[nid] == 0) N++;
                    }
                    // down
                    if (j > 0)
                    {
                        int nid = IU(i, j - 1);
                        sumN += u[nid];
                        if (solidU[nid] == 0) N++;
                    }
                    // up
                    if (j < NY - 1)
                    {
                        int nid = IU(i, j + 1);
                        sumN += u[nid];
                        if (solidU[nid] == 0) N++;
                    }

                    float denom = 1.0f + alpha * MathF.Max(1, N);
                    uTmp[id] = (u0[id] + alpha * sumN) / denom;
                }
            (u, uTmp) = (uTmp, u);

            // v
            for (int j = 0; j <= NY; j++)
                for (int i = 0; i < NX; i++)
                {
                    int id = IV(i, j);
                    if (solidV[id] != 0) { vTmp[id] = v0[id]; continue; }

                    float sumN = 0f; int N = 0;
                    // left
                    if (i > 0)
                    {
                        int nid = IV(i - 1, j);
                        sumN += v[nid];
                        if (solidV[nid] == 0) N++;
                    }
                    // right
                    if (i < NX - 1)
                    {
                        int nid = IV(i + 1, j);
                        sumN += v[nid];
                        if (solidV[nid] == 0) N++;
                    }
                    // down
                    if (j > 0)
                    {
                        int nid = IV(i, j - 1);
                        sumN += v[nid];
                        if (solidV[nid] == 0) N++;
                    }
                    // up
                    if (j < NY)
                    {
                        int nid = IV(i, j + 1);
                        sumN += v[nid];
                        if (solidV[nid] == 0) N++;
                    }

                    float denom = 1.0f + alpha * MathF.Max(1, N);
                    vTmp[id] = (v0[id] + alpha * sumN) / denom;
                }
            (v, vTmp) = (vTmp, v);

            EnforceBoundaries();
            ApplySolidFaceBCs(circleVel);
        }
    }

    // ============================================================
    //                 DIVERGENCE & RHS (with solid)
    // ============================================================
    static void ComputeDivergenceAndRHS(float dt)
    {
        // kept for compatibility (not used)
        ComputeDivergenceAndRHS_WithSolid(dt);
    }

    static void ComputeDivergenceAndRHS_WithSolid(float dt)
    {
        float scale = RHO / dt;
        for (int j = 0; j < NY; j++)
            for (int i = 0; i < NX; i++)
            {
                float ue = (solidU[IU(i + 1, j)] == 0) ? u[IU(i + 1, j)] : circleVel.X;
                float uw = (solidU[IU(i, j)] == 0) ? u[IU(i, j)] : circleVel.X;
                float vn = (solidV[IV(i, j + 1)] == 0) ? v[IV(i, j + 1)] : circleVel.Y;
                float vs = (solidV[IV(i, j)] == 0) ? v[IV(i, j)] : circleVel.Y;
                float div = (ue - uw) + (vn - vs); // h = 1
                divergence[Idx(i, j)] = div;
                rhs[Idx(i, j)] = scale * div;
            }
        // anchor rhs cleared later in solver
    }

    // -------- Stage 2: Conjugate Gradient pressure solver (with solid-aware ApplyL) --------
    static void PrecomputeCGDiagonal()
    {
        // basic (no-solid) diag; will be updated per-frame once masks exist
        for (int j = 0; j < NY; j++)
            for (int i = 0; i < NX; i++)
            {
                int N = 0;
                if (i > 0) N++;
                if (i < NX - 1) N++;
                if (j > 0) N++;
                if (j < NY - 1) N++;
                cg_diagInv[Idx(i, j)] = (N > 0) ? 1.0f / N : 1.0f;
            }
        cg_diagInv[0] = 1.0f; // initial anchor
    }

    // Recompute diagonal inverse from current solid masks (call per-frame after BuildLevelSet)
    static void UpdateCGDiagonalFromMask()
    {
        // find anchor = first fluid cell
        anchorIndex = 0;
        for (int k = 0; k < NX * NY; k++) { if (solidCell[k] == 0) { anchorIndex = k; break; } }

        for (int j = 0; j < NY; j++)
            for (int i = 0; i < NX; i++)
            {
                int id = Idx(i, j);
                if (id == anchorIndex) { cg_diagInv[id] = 1.0f; continue; }

                float sumA = 0f;
                // east neighbor allowed if east face open
                if (i < NX - 1 && solidU[IU(i + 1, j)] == 0) sumA += 1f;
                // west neighbor
                if (i > 0 && solidU[IU(i, j)] == 0) sumA += 1f;
                // north neighbor
                if (j < NY - 1 && solidV[IV(i, j + 1)] == 0) sumA += 1f;
                // south neighbor
                if (j > 0 && solidV[IV(i, j)] == 0) sumA += 1f;

                if (sumA <= 0f) cg_diagInv[id] = 1.0f;
                else cg_diagInv[id] = 1.0f / sumA;
            }
    }

    static (int iters, float relRes) SolvePressureCG(int maxIters, float relTol)
    {
        // Anchor rhs at 0 — consistent with ApplyL
        rhs[anchorIndex] = 0.0f;

        // r = b - A p
        ApplyL(pressure, cg_Ap);
        float bnorm = 0f;
        for (int k = 0; k < NX * NY; k++)
        {
            float r = rhs[k] - cg_Ap[k];
            cg_r[k] = r;
            cg_z[k] = cg_diagInv[k] * r; // Jacobi preconditioner (diagonal)
            cg_d[k] = cg_z[k];
            bnorm += rhs[k] * rhs[k];
        }
        bnorm = MathF.Sqrt(bnorm);
        if (bnorm < 1e-20f) return (0, 0f); // already satisfied

        float rz_old = Dot(cg_r, cg_z);
        int it;
        float rel = float.NaN;

        for (it = 0; it < maxIters; it++)
        {
            ApplyL(cg_d, cg_Ap);

            float denom = Dot(cg_d, cg_Ap);
            if (MathF.Abs(denom) < 1e-20f) break;
            float alpha = rz_old / denom;

            Saxpy(pressure, cg_d, alpha);     // p += α d
            Saxpy(cg_r, cg_Ap, -alpha);       // r -= α A d

            // Check residual
            float rnorm = Norm(cg_r);
            rel = rnorm / bnorm;
            if (rel <= relTol) { it++; break; }

            // z = M^{-1} r  (Jacobi)
            for (int k = 0; k < NX * NY; k++) cg_z[k] = cg_diagInv[k] * cg_r[k];

            float rz_new = Dot(cg_r, cg_z);
            float beta = rz_new / (rz_old + 1e-30f);

            // d = z + β d
            for (int k = 0; k < NX * NY; k++) cg_d[k] = cg_z[k] + beta * cg_d[k];

            rz_old = rz_new;
        }

        return (it, rel);
    }

    // L p = sum(open-neighbor p) - sum(open-neighbors) * p
    // using anchorIndex to fix nullspace
    static void ApplyL(float[] x, float[] Lx)
    {
        for (int j = 0; j < NY; j++)
            for (int i = 0; i < NX; i++)
            {
                int id = Idx(i, j);
                if (id == anchorIndex) { Lx[id] = x[id]; continue; }

                float sum = 0f; int N = 0;
                // west neighbor (face IU(i,j) between (i-1) and i)
                if (i > 0 && solidU[IU(i, j)] == 0) { sum += x[Idx(i - 1, j)]; N++; }
                // east neighbor
                if (i < NX - 1 && solidU[IU(i + 1, j)] == 0) { sum += x[Idx(i + 1, j)]; N++; }
                // south neighbor
                if (j > 0 && solidV[IV(i, j)] == 0) { sum += x[Idx(i, j - 1)]; N++; }
                // north neighbor
                if (j < NY - 1 && solidV[IV(i, j + 1)] == 0) { sum += x[Idx(i, j + 1)]; N++; }

                Lx[id] = sum - N * x[id];
            }
    }

    static float Dot(float[] a, float[] b)
    {
        double acc = 0.0;
        int n = a.Length;
        for (int i = 0; i < n; i++) acc += (double)a[i] * b[i];
        return (float)acc;
    }
    static float Norm(float[] a) => MathF.Sqrt(Dot(a, a));
    static void Saxpy(float[] y, float[] x, float alpha)
    {
        int n = y.Length;
        for (int i = 0; i < n; i++) y[i] += alpha * x[i];
    }

    // ============================================================
    //                 PROJECTION (with solids)
    // ============================================================
    static void ProjectVelocities(float dt) => ProjectVelocities_WithSolid(dt);

    static void ProjectVelocities_WithSolid(float dt)
    {
        float scale = dt / RHO;
        // u faces: use pressure differences in x but only if face open
        for (int j = 0; j < NY; j++)
            for (int i = 1; i < NX; i++)
            {
                int idu = IU(i, j);
                if (solidU[idu] != 0)
                {
                    u[idu] = circleVel.X; // enforce solid face velocity
                    continue;
                }
                float gradp = pressure[Idx(i, j)] - pressure[Idx(i - 1, j)];
                u[idu] -= scale * gradp;
            }
        // v faces: use pressure differences in y
        for (int j = 1; j < NY; j++)
            for (int i = 0; i < NX; i++)
            {
                int idv = IV(i, j);
                if (solidV[idv] != 0)
                {
                    v[idv] = circleVel.Y;
                    continue;
                }
                float gradp = pressure[Idx(i, j)] - pressure[Idx(i, j - 1)];
                v[idv] -= scale * gradp;
            }
    }

    static void ComputeDivergencePost()
    {
        for (int j = 0; j < NY; j++)
            for (int i = 0; i < NX; i++)
            {
                float dudx = u[IU(i + 1, j)] - u[IU(i, j)];
                float dvdy = v[IV(i, j + 1)] - v[IV(i, j)];
                divergencePost[Idx(i, j)] = dudx + dvdy;
            }
    }

    // ============================================================
    //                   VELOCITY INJECTION (mouse)
    // ============================================================
    static void InjectVelocity(Vector2 gridPos, Vector2 impulse)
    {
        // u field
        AddBilinearFace(u, NX + 1, NY, gridPos.X, gridPos.Y - 0.5f, impulse.X);
        AddBilinearFace(u, NX + 1, NY, gridPos.X - 1, gridPos.Y - 0.5f, impulse.X);
        AddBilinearFace(u, NX + 1, NY, gridPos.X + 1, gridPos.Y - 0.5f, impulse.X);
        AddBilinearFace(u, NX + 1, NY, gridPos.X, gridPos.Y + 0.5f, impulse.X);
        AddBilinearFace(u, NX + 1, NY, gridPos.X, gridPos.Y - 1.5f, impulse.X);

        // v field
        AddBilinearFace(v, NX, NY + 1, gridPos.X - 0.5f, gridPos.Y, impulse.Y);
        AddBilinearFace(v, NX, NY + 1, gridPos.X - 0.5f, gridPos.Y - 1, impulse.Y);
        AddBilinearFace(v, NX, NY + 1, gridPos.X - 0.5f, gridPos.Y + 1, impulse.Y);
        AddBilinearFace(v, NX, NY + 1, gridPos.X + 0.5f, gridPos.Y, impulse.Y);
        AddBilinearFace(v, NX, NY + 1, gridPos.X - 1.5f, gridPos.Y, impulse.Y);
    }

    static void AddBilinearFace(float[] A, int SX, int SY, float x, float y, float value)
    {
        const float EPS = 1e-4f;
        float xc = MathF.Max(0.0f, MathF.Min(x, SX - 1 - EPS));
        float yc = MathF.Max(0.0f, MathF.Min(y, SY - 1 - EPS));

        int i0 = (int)MathF.Floor(xc);
        int j0 = (int)MathF.Floor(yc);
        int i1 = Math.Min(i0 + 1, SX - 1);
        int j1 = Math.Min(j0 + 1, SY - 1);

        float sx = xc - i0;
        float sy = yc - j0;

        float w00 = (1 - sx) * (1 - sy);
        float w10 = sx * (1 - sy);
        float w01 = (1 - sx) * sy;
        float w11 = sx * sy;

        int idx00 = i0 + j0 * SX;
        int idx10 = i1 + j0 * SX;
        int idx01 = i0 + j1 * SX;
        int idx11 = i1 + j1 * SX;

        A[idx00] += value * w00;
        A[idx10] += value * w10;
        A[idx01] += value * w01;
        A[idx11] += value * w11;
    }

    // ============================================================
    //                          SOLID LEVELSET & MASKS (Level A)
    // ============================================================
    static void BuildLevelSetCircle(Vector2 C, float R)
    {
        float cx = C.X, cy = C.Y, r = R;
        // cells
        for (int j = 0; j < NY; j++)
            for (int i = 0; i < NX; i++)
            {
                int id = Idx(i, j);
                float px = i + 0.5f;
                float py = j + 0.5f;
                float d = MathF.Sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy)) - r;
                phiCell[id] = d;
                solidCell[id] = (byte)(d < 0f ? 1 : 0);
            }
        // u faces (i, j+0.5)
        for (int j = 0; j < NY; j++)
            for (int i = 0; i <= NX; i++)
            {
                int id = IU(i, j);
                float px = i;
                float py = j + 0.5f;
                float d = MathF.Sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy)) - r;
                phiU[id] = d;
                solidU[id] = (byte)(d < 0f ? 1 : 0);
            }
        // v faces (i+0.5, j)
        for (int j = 0; j <= NY; j++)
            for (int i = 0; i < NX; i++)
            {
                int id = IV(i, j);
                float px = i + 0.5f;
                float py = j;
                float d = MathF.Sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy)) - r;
                phiV[id] = d;
                solidV[id] = (byte)(d < 0f ? 1 : 0);
            }
    }

    static void ApplySolidFaceBCs(Vector2 Ub)
    {
        // set faces inside solid to object normal velocity (free-slip: set normal)
        for (int j = 0; j < NY; j++)
            for (int i = 0; i <= NX; i++)
            {
                int id = IU(i, j);
                if (solidU[id] != 0) u[id] = Ub.X;
            }
        for (int j = 0; j <= NY; j++)
            for (int i = 0; i < NX; i++)
            {
                int id = IV(i, j);
                if (solidV[id] != 0) v[id] = Ub.Y;
            }
    }

    static void ZeroDensityInSolidCells()
    {
        for (int k = 0; k < NX * NY; k++) if (solidCell[k] != 0) density[k] = 0f;
    }

    // ============================================================
    //                      SAFETY / DIAGNOSTICS
    // ============================================================
    static bool ScanForNaNInf()
    {
        for (int k = 0; k < NX * NY; k++)
            if (float.IsNaN(pressure[k]) || float.IsInfinity(pressure[k]) ||
                float.IsNaN(density[k]) || float.IsInfinity(density[k]))
                return true;

        for (int k = 0; k < NU; k++)
            if (float.IsNaN(u[k]) || float.IsInfinity(u[k])) return true;

        for (int k = 0; k < NV; k++)
            if (float.IsNaN(v[k]) || float.IsInfinity(v[k])) return true;

        return false;
    }

    // ============================================================
    //                          DRAW
    // ============================================================
    static void Draw()
    {
        Raylib.BeginDrawing();
        Raylib.ClearBackground(Color.Black);

        // Stage 1: update density texture and draw once
        UpdateDensityTexture();
        Rectangle src = new Rectangle(0, 0, NX, NY);
        Rectangle dst = new Rectangle(0, 0, NX * CELL_SIZE, NY * CELL_SIZE);
        Raylib.DrawTexturePro(densityTex, src, dst, new Vector2(0, 0), 0.0f, Color.White);

        if (showVel) DrawVelocityField(6);

        // draw object (translucent fill + outline)
        int cx = (int)(circleCenter.X * CELL_SIZE);
        int cy = (int)(circleCenter.Y * CELL_SIZE);
        float rpx = circleRadius * CELL_SIZE;
        Raylib.DrawCircle(cx, cy, rpx, new Color(60, 140, 220, 40)); // translucent fill
        Raylib.DrawCircleLines(cx, cy, rpx, Color.SkyBlue); // outline

        // Diagnostics overlay & mini divergence preview
        diag.DrawOverlay();
        DrawDivergencePreview();

        Raylib.EndDrawing();
    }

    static void UpdateDensityTexture()
    {
        for (int j = 0; j < NY; j++)
            for (int i = 0; i < NX; i++)
            {
                int id = Idx(i, j);
                byte c = (byte)(Math.Clamp(density[id], 0.0f, 1.0f) * 255);
                int p = id * 4;
                pixels[p + 0] = c;
                pixels[p + 1] = c;
                pixels[p + 2] = c;
                pixels[p + 3] = 255;
            }
        Raylib.UpdateTexture(densityTex, pixels);
    }

    static void DrawVelocityField(int stride)
    {
        for (int j = 0; j < NY; j += stride)
            for (int i = 0; i < NX; i += stride)
            {
                Vector2 pos = new Vector2((i + 0.5f) * CELL_SIZE, (j + 0.5f) * CELL_SIZE);
                Vector2 vel = SampleVelocity(new Vector2(i + 0.5f, j + 0.5f));
                Vector2 tip = pos + vel * CELL_SIZE * 0.2f;
                Raylib.DrawLine((int)pos.X, (int)pos.Y, (int)tip.X, (int)tip.Y, Color.DarkGreen);
            }
    }

    static void DrawDivergencePreview()
    {
        int previewSize = 128;
        int px = (int)(NX * CELL_SIZE) - previewSize - 10;
        int py = 10;
        int cell = Math.Max(1, previewSize / NX);

        Raylib.DrawRectangle(px - 2, py - 2, previewSize + 4, previewSize + 4, Color.DarkGray);

        for (int pyi = 0; pyi < previewSize; pyi += cell)
            for (int pxi = 0; pxi < previewSize; pxi += cell)
            {
                int gi = (int)((float)pxi / previewSize * NX);
                int gj = (int)((float)pyi / previewSize * NY);
                gi = Math.Clamp(gi, 0, NX - 1);
                gj = Math.Clamp(gj, 0, NY - 1);
                float d = divergencePost[Idx(gi, gj)];
                float s = MathF.Tanh(MathF.Abs(d) * 10.0f);
                Color col = d > 0 ? ColorLerp(Color.Black, Color.Red, s) : ColorLerp(Color.Black, Color.SkyBlue, s);
                Raylib.DrawRectangle(px + pxi, py + pyi, cell, cell, col);
            }
    }

    static Color ColorLerp(Color a, Color b, float t)
    {
        t = MathF.Max(0, MathF.Min(1, t));
        return new Color(
            (byte)(a.R + (b.R - a.R) * t),
            (byte)(a.G + (b.G - a.G) * t),
            (byte)(a.B + (b.B - a.B) * t),
            (byte)255
        );
    }

    // ============================================================
    //                        DIAGNOSTICS
    // ============================================================
    class Diagnostics
    {
        const int LEN = 300;
        int idx = 0;
        int filled = 0;

        float[] massHist = new float[LEN];
        float[] tvHist = new float[LEN];
        float[] keHist = new float[LEN];
        float[] divPreMaxHist = new float[LEN];
        float[] divPostMaxHist = new float[LEN];
        float[] presResHist = new float[LEN];
        float[] cflHist = new float[LEN];
        float[] fpsHist = new float[LEN];
        float[] cgItersHist = new float[LEN];

        public float lastMass = 0, lastTV = 0, lastKE = 0, lastMaxSpeed = 0;
        public float lastDivPreMax = 0, lastDivPostMax = 0;
        public float lastPressureResidual = 0, lastCFL = 0;
        public int lastCGIters = 0;
        public int frames = 0;

        bool show = true;
        int left = 8, top = 8;

        public void toggleShow() { show = !show; }
        public void ClearHistory()
        {
            idx = 0; filled = 0;
            Array.Clear(massHist); Array.Clear(tvHist); Array.Clear(keHist);
            Array.Clear(divPreMaxHist); Array.Clear(divPostMaxHist); Array.Clear(presResHist);
            Array.Clear(cflHist); Array.Clear(fpsHist); Array.Clear(cgItersHist);
        }

        public void GatherAll(
            float[] density,
            float[] u, float[] v,
            float[] divPre, float[] divPost,
            float[] pressure,
            float pressureResidual,
            float dt,
            long msAdvection,
            long msForces,
            long msDiffuse,
            long msDivergence,
            long msPressure,
            long msProject,
            long msBoundaries,
            long msFrame,
            int cgIters)
        {
            // Mass & TV
            float mass = 0f, tv = 0f;
            for (int j = 0; j < NY; j++)
                for (int i = 0; i < NX; i++)
                {
                    int id = Idx(i, j);
                    float d = density[id];
                    mass += d;
                    if (i + 1 < NX) tv += MathF.Abs(density[Idx(i + 1, j)] - d);
                    if (j + 1 < NY) tv += MathF.Abs(density[Idx(i, j + 1)] - d);
                }

            // KE & max speed
            float ke = 0f, maxSpeed = 0f;
            for (int j = 0; j < NY; j++)
                for (int i = 0; i <= NX; i++)
                {
                    float val = u[IU(i, j)];
                    ke += 0.5f * val * val;
                    if (MathF.Abs(val) > maxSpeed) maxSpeed = MathF.Abs(val);
                }
            for (int j = 0; j <= NY; j++)
                for (int i = 0; i < NX; i++)
                {
                    float val = v[IV(i, j)];
                    ke += 0.5f * val * val;
                    if (MathF.Abs(val) > maxSpeed) maxSpeed = MathF.Abs(val);
                }

            // Divergence norms (pre/post)
            float divPreMax = 0f, divPostMax = 0f;
            for (int k = 0; k < NX * NY; k++)
            {
                float a = divPre[k], b = divPost[k];
                if (MathF.Abs(a) > divPreMax) divPreMax = MathF.Abs(a);
                if (MathF.Abs(b) > divPostMax) divPostMax = MathF.Abs(b);
            }

            float cfl = maxSpeed * dt;

            lastMass = mass; lastTV = tv; lastKE = ke; lastMaxSpeed = maxSpeed;
            lastDivPreMax = divPreMax; lastDivPostMax = divPostMax;
            lastPressureResidual = pressureResidual; lastCFL = cfl; lastCGIters = cgIters;

            massHist[idx] = mass;
            tvHist[idx] = tv;
            keHist[idx] = ke;
            divPreMaxHist[idx] = divPreMax;
            divPostMaxHist[idx] = divPostMax;
            presResHist[idx] = pressureResidual;
            cflHist[idx] = cfl;
            fpsHist[idx] = Raylib.GetFPS();
            cgItersHist[idx] = cgIters;

            idx = (idx + 1) % LEN;
            filled = Math.Min(filled + 1, LEN);
            frames++;
        }

        public void DrawOverlay()
        {
            if (!show) return;

            int x = left, y = top, lineh = 16;
            Raylib.DrawRectangle(x - 6, y - 6, 480, 240, new Color(0, 0, 0, 150));

            Raylib.DrawText("Diagnostics (D toggles) — P: pause, N: step, C: clear, V: vel", x, y, 12, Color.LightGray);
            y += lineh;

            Raylib.DrawText($"Frame: {frames}", x, y, 12, Color.LightGray); y += lineh;
            Raylib.DrawText($"Mass: {lastMass:F3}   TV: {lastTV:F3}   KE: {lastKE:F3}", x, y, 12, Color.LightGray); y += lineh;
            Raylib.DrawText($"MaxSpeed: {lastMaxSpeed:F3}   CFL: {lastCFL:F3}", x, y, 12, Color.LightGray); y += lineh;
            Raylib.DrawText($"DivPreMax: {lastDivPreMax:E3}   DivPostMax: {lastDivPostMax:E3}", x, y, 12, Color.LightGray); y += lineh;
            Raylib.DrawText($"CG: iters {lastCGIters}  residual {lastPressureResidual:E3}", x, y, 12, Color.LightGray); y += lineh;

            DrawSparkline("mass", massHist, x, y, 220, 40, filled, idx); y += 44;
            DrawSparkline("KE", keHist, x, y, 220, 40, filled, idx); y += 44;
            DrawSparkline("divPostMax", divPostMaxHist, x, y, 220, 40, filled, idx); y += 44;
        }

        void DrawSparkline(string label, float[] buffer, int x, int y, int w, int h, int filled, int idx)
        {
            Raylib.DrawText(label, x, y, 10, Color.LightGray);
            int bx = x + 60, by = y;
            Raylib.DrawRectangle(bx - 2, by - 2, w + 4, h + 4, new Color(80, 80, 80, 120));
            if (filled == 0) return;

            float min = float.MaxValue, max = float.MinValue;
            for (int i = 0; i < filled; i++)
            {
                float v = buffer[i];
                if (float.IsNaN(v) || float.IsInfinity(v)) continue;
                if (v < min) min = v;
                if (v > max) max = v;
            }
            if (min == float.MaxValue) { min = 0; max = 1; }
            if (min == max) max = min + 1e-6f;

            int display = Math.Min(filled, w);
            for (int i = 0; i < display - 1; i++)
            {
                int a = (idx - i - 1 + LEN) % LEN;
                int b = (idx - i - 2 + LEN) % LEN;
                float va = buffer[a], vb = buffer[b];
                float na = (va - min) / (max - min);
                float nb = (vb - min) / (max - min);
                int xa = bx + (display - 1 - i);
                int xb = bx + (display - 2 - i);
                int ya = by + h - (int)(na * h);
                int yb = by + h - (int)(nb * h);
                Raylib.DrawLine(xa, ya, xb, yb, Color.Green);
            }
        }
    }
}
