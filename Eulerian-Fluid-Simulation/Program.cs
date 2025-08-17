using Raylib_cs;
using System;
using System.Numerics;
using System.Diagnostics;

public class Program
{
    // Grid dimensions
    const int GRID_WIDTH = 128;
    const int GRID_HEIGHT = 128;
    const float CELL_SIZE = 6.0f; // Pixels per cell (rendering only)

    // Simulation step (seconds). We'll keep it fixed for now; later we can compute CFL.
    const float DT = 0.1f;

    // Fluid fields (cell-centered scalars)
    static float[,] pressure = new float[GRID_WIDTH, GRID_HEIGHT];
    static float[,] density = new float[GRID_WIDTH, GRID_HEIGHT];

    // Staggered MAC velocities
    // u: horizontal component at vertical faces  -> (Nx+1, Ny)
    // v: vertical component at horizontal faces  -> (Nx, Ny+1)
    static float[,] u = new float[GRID_WIDTH + 1, GRID_HEIGHT];
    static float[,] v = new float[GRID_WIDTH, GRID_HEIGHT + 1];

    // Temporary buffers for advection (reuse every frame; no GC churn)
    static float[,] densityTmp = new float[GRID_WIDTH, GRID_HEIGHT];
    static float[,] uTmp = new float[GRID_WIDTH + 1, GRID_HEIGHT];
    static float[,] vTmp = new float[GRID_WIDTH, GRID_HEIGHT + 1];

    // ===== Physics parameters (tweakable) =====
    const float RHO = 1.0f;              // fluid density
    const float GRAVITY_X = 0.0f;        // body force x
    const float GRAVITY_Y = -9.0f;       // body force y (down)
    const float VISCOSITY = 0.0005f;     // kinematic viscosity (set 0 to disable)
    const int DIFFUSION_ITERS = 10;      // Jacobi iterations for diffusion
    const int PRESSURE_ITERS = 200;      // Jacobi iterations for Poisson

    // Poisson / divergence buffers (cell-centered)
    static float[,] pressureTmp = new float[GRID_WIDTH, GRID_HEIGHT];
    static float[,] rhs = new float[GRID_WIDTH, GRID_HEIGHT];
    static float[,] divergence = new float[GRID_WIDTH, GRID_HEIGHT];
    static float[,] divergencePost = new float[GRID_WIDTH, GRID_HEIGHT];

    // Diffusion RHS copies (staggered)
    static float[,] u0 = new float[GRID_WIDTH + 1, GRID_HEIGHT];
    static float[,] v0 = new float[GRID_WIDTH, GRID_HEIGHT + 1];

    // Mouse state for simple velocity injection
    static Vector2 prevMouse;
    const float MOUSE_FORCE_SCALE = 15.0f; // tweakable impulse strength

    // Diagnostics
    static Diagnostics diag = new Diagnostics();

    // Control
    static bool paused = false;
    static bool singleStep = false;

    public static void Main()
    {
        int screenWidth = (int)(GRID_WIDTH * CELL_SIZE);
        int screenHeight = (int)(GRID_HEIGHT * CELL_SIZE);

        Raylib.InitWindow(screenWidth, screenHeight, "2D Eulerian Fluid Simulation - MAC Grid (Advection+Projection+Diagnostics)");
        Raylib.SetTargetFPS(120);
        prevMouse = Raylib.GetMousePosition();

        while (!Raylib.WindowShouldClose())
        {
            // Controls
            if (Raylib.IsKeyPressed(KeyboardKey.D)) diag.toggleShow();
            if (Raylib.IsKeyPressed(KeyboardKey.P)) paused = !paused;
            if (Raylib.IsKeyPressed(KeyboardKey.N)) singleStep = true;
            if (Raylib.IsKeyPressed(KeyboardKey.C)) diag.ClearHistory(); // clear time-series

            if (!paused || singleStep)
            {
                Update();
                singleStep = false;
            }

            Draw();
        }

        Raylib.CloseWindow();
    }

    static void Update()
    {
        var swFrame = Stopwatch.StartNew();

        float dt = DT; // fixed for now

        // Paint density in a small radius around mouse
        if (Raylib.IsMouseButtonDown(MouseButton.Left))
        {
            Vector2 mouse = Raylib.GetMousePosition();
            int gx = (int)(mouse.X / CELL_SIZE);
            int gy = (int)(mouse.Y / CELL_SIZE);

            int radius = 6; // in grid cells

            for (int y = gy - radius; y <= gy + radius; y++)
            {
                for (int x = gx - radius; x <= gx + radius; x++)
                {
                    if (x >= 0 && x < GRID_WIDTH && y >= 0 && y < GRID_HEIGHT)
                    {
                        int dx = x - gx;
                        int dy = y - gy;
                        float distSq = dx * dx + dy * dy;
                        if (distSq <= radius * radius)
                        {
                            // deposit proportional to distance (feels nicer)
                            float falloff = 1.0f - MathF.Sqrt(distSq) / (radius + 1e-6f);
                            density[x, y] = MathF.Min(1.0f, density[x, y] + 1.0f * falloff);
                        }
                    }
                }
            }
        }

        // Simple velocity injection with right mouse drag (adds momentum near cursor)
        Vector2 mouseNow = Raylib.GetMousePosition();
        if (Raylib.IsMouseButtonDown(MouseButton.Right))
        {
            Vector2 dm = (mouseNow - prevMouse) / CELL_SIZE; // convert to grid units per frame
            Vector2 gridPos = new Vector2(mouseNow.X / CELL_SIZE, mouseNow.Y / CELL_SIZE);
            InjectVelocity(gridPos, dm * MOUSE_FORCE_SCALE);
        }
        prevMouse = mouseNow;

        var swAdvection = Stopwatch.StartNew();
        // --- Core fluid step order ---
        // 1) Advect scalars and velocity by current velocity
        AdvectDensity(dt);
        AdvectVelocity(dt);
        swAdvection.Stop();

        var swForces = Stopwatch.StartNew();
        // 2) Apply body forces (e.g., gravity)
        ApplyForces(dt);
        swForces.Stop();

        var swDiffuse = Stopwatch.StartNew();
        // 3) Implicit diffusion (viscosity)
        if (VISCOSITY > 0.0f)
            DiffuseVelocities(dt);
        swDiffuse.Stop();

        var swBoundary1 = Stopwatch.StartNew();
        // 4) Enforce simple box boundary before projection
        EnforceBoundaries();
        swBoundary1.Stop();

        var swDivergence = Stopwatch.StartNew();
        // 5) Build divergence and RHS for pressure Poisson
        ComputeDivergenceAndRHS(dt);
        swDivergence.Stop();

        var swPressure = Stopwatch.StartNew();
        // 6) Solve for pressure (Jacobi)
        float lastPressureResidual = SolvePressureJacobi(PRESSURE_ITERS);
        swPressure.Stop();

        var swProject = Stopwatch.StartNew();
        // 7) Subtract pressure gradient (projection)
        ProjectVelocities(dt);
        swProject.Stop();

        var swBoundary2 = Stopwatch.StartNew();
        // 8) Final boundary clamp
        EnforceBoundaries();
        swBoundary2.Stop();

        swFrame.Stop();

        // compute post-projection divergence for diagnostics
        ComputeDivergencePost();

        // NaN / Inf check
        bool hasNaN = ScanForNaNInf();
        if (hasNaN)
        {
            paused = true;
            Console.WriteLine("NaN/Inf detected — simulation paused. Check arrays.");
        }

        // Gather diagnostics (fills ring buffers etc.)
        diag.GatherAll(
            density, u, v,
            divergence, divergencePost,
            pressure, lastPressureResidual,
            dt,
            swAdvection.ElapsedMilliseconds,
            swForces.ElapsedMilliseconds,
            swDiffuse.ElapsedMilliseconds,
            swDivergence.ElapsedMilliseconds,
            swPressure.ElapsedMilliseconds,
            swProject.ElapsedMilliseconds,
            swBoundary1.ElapsedMilliseconds + swBoundary2.ElapsedMilliseconds,
            swFrame.ElapsedMilliseconds
        );
    }

    // ===================== Advection (Semi-Lagrangian, RK2) =====================

    static void AdvectDensity(float dt)
    {
        // For each cell center (i+0.5, j+0.5), backtrace and sample previous density
        for (int j = 0; j < GRID_HEIGHT; j++)
        {
            for (int i = 0; i < GRID_WIDTH; i++)
            {
                Vector2 x = new Vector2(i + 0.5f, j + 0.5f);
                Vector2 v1 = SampleVelocity(x);
                Vector2 xmid = x - 0.5f * dt * v1;
                Vector2 v2 = SampleVelocity(xmid);
                Vector2 xprev = x - dt * v2;
                xprev = ClampToScalarDomain(xprev);
                densityTmp[i, j] = SampleScalar(density, xprev);
            }
        }
        // swap: write back into density
        Swap(ref density, ref densityTmp);
    }

    static void AdvectVelocity(float dt)
    {
        // Advect u on its staggered grid: positions (i, j+0.5), i in [0..Nx], j in [0..Ny-1]
        for (int j = 0; j < GRID_HEIGHT; j++)
        {
            for (int i = 0; i <= GRID_WIDTH; i++)
            {
                Vector2 x = new Vector2(i, j + 0.5f);
                Vector2 v1 = SampleVelocity(x);
                Vector2 xmid = x - 0.5f * dt * v1;
                Vector2 v2 = SampleVelocity(xmid);
                Vector2 xprev = x - dt * v2;
                xprev = ClampToUDomain(xprev);
                uTmp[i, j] = SampleU(u, xprev);
            }
        }

        // Advect v on its staggered grid: positions (i+0.5, j), i in [0..Nx-1], j in [0..Ny]
        for (int j = 0; j <= GRID_HEIGHT; j++)
        {
            for (int i = 0; i < GRID_WIDTH; i++)
            {
                Vector2 x = new Vector2(i + 0.5f, j);
                Vector2 v1 = SampleVelocity(x);
                Vector2 xmid = x - 0.5f * dt * v1;
                Vector2 v2 = SampleVelocity(xmid);
                Vector2 xprev = x - dt * v2;
                xprev = ClampToVDomain(xprev);
                vTmp[i, j] = SampleV(v, xprev);
            }
        }

        Swap(ref u, ref uTmp);
        Swap(ref v, ref vTmp);
    }

    // ===================== Sampling helpers (bilinear on staggered grids) =====================

    static Vector2 SampleVelocity(Vector2 pos)
    {
        float ux = SampleU(u, pos);
        float vy = SampleV(v, pos);
        return new Vector2(ux, vy);
    }

    static float SampleScalar(float[,] A, Vector2 pos)
    {
        float xs = pos.X - 0.5f; // cell-centered index space
        float ys = pos.Y - 0.5f;
        return Bilinear(A, xs, ys, GRID_WIDTH, GRID_HEIGHT);
    }

    static float SampleU(float[,] U, Vector2 pos)
    {
        // U lives on vertical faces at (i, j+0.5), array size (Nx+1, Ny)
        float xu = pos.X;         // index x maps directly
        float yu = pos.Y - 0.5f;  // shift to face index space
        return Bilinear(U, xu, yu, GRID_WIDTH + 1, GRID_HEIGHT);
    }

    static float SampleV(float[,] V, Vector2 pos)
    {
        // V lives on horizontal faces at (i+0.5, j), array size (Nx, Ny+1)
        float xv = pos.X - 0.5f;
        float yv = pos.Y;
        return Bilinear(V, xv, yv, GRID_WIDTH, GRID_HEIGHT + 1);
    }

    static float Bilinear(float[,] A, float x, float y, int NX, int NY)
    {
        const float EPS = 1e-4f;
        float xc = MathF.Max(0.0f, MathF.Min(x, NX - 1 - EPS));
        float yc = MathF.Max(0.0f, MathF.Min(y, NY - 1 - EPS));

        int i0 = (int)MathF.Floor(xc);
        int j0 = (int)MathF.Floor(yc);
        int i1 = i0 + 1;
        int j1 = j0 + 1;

        float sx = xc - i0;
        float sy = yc - j0;

        if (i1 >= NX) i1 = NX - 1;
        if (j1 >= NY) j1 = NY - 1;

        float a00 = A[i0, j0];
        float a10 = A[i1, j0];
        float a01 = A[i0, j1];
        float a11 = A[i1, j1];

        float v0 = a00 * (1 - sx) + a10 * sx;
        float v1 = a01 * (1 - sx) + a11 * sx;
        return v0 * (1 - sy) + v1 * sy;
    }

    // ===================== Domains & utilities =====================

    static Vector2 ClampToScalarDomain(Vector2 p)
    {
        float x = MathF.Max(0.5f, MathF.Min(p.X, GRID_WIDTH - 0.5f));
        float y = MathF.Max(0.5f, MathF.Min(p.Y, GRID_HEIGHT - 0.5f));
        return new Vector2(x, y);
    }

    static Vector2 ClampToUDomain(Vector2 p)
    {
        float x = MathF.Max(0.0f, MathF.Min(p.X, GRID_WIDTH));
        float y = MathF.Max(0.5f, MathF.Min(p.Y, GRID_HEIGHT - 0.5f));
        return new Vector2(x, y);
    }

    static Vector2 ClampToVDomain(Vector2 p)
    {
        float x = MathF.Max(0.5f, MathF.Min(p.X, GRID_WIDTH - 0.5f));
        float y = MathF.Max(0.0f, MathF.Min(p.Y, GRID_HEIGHT));
        return new Vector2(x, y);
    }

    static void Swap(ref float[,] A, ref float[,] B)
    {
        var tmp = A; A = B; B = tmp;
    }

    // ===================== Forces, diffusion, projection =====================

    static void ApplyForces(float dt)
    {
        // Add uniform body force to velocities
        for (int j = 0; j < GRID_HEIGHT; j++)
        {
            for (int i = 0; i <= GRID_WIDTH; i++)
                u[i, j] += dt * GRAVITY_X;
        }
        for (int j = 0; j <= GRID_HEIGHT; j++)
        {
            for (int i = 0; i < GRID_WIDTH; i++)
                v[i, j] += dt * GRAVITY_Y;
        }
    }

    static void EnforceBoundaries()
    {
        // No-flow box: zero normal velocity at domain edges
        for (int j = 0; j < GRID_HEIGHT; j++)
        {
            u[0, j] = 0.0f;
            u[GRID_WIDTH, j] = 0.0f;
        }
        for (int i = 0; i < GRID_WIDTH; i++)
        {
            v[i, 0] = 0.0f;
            v[i, GRID_HEIGHT] = 0.0f;
        }
    }

    static void DiffuseVelocities(float dt)
    {
        float alpha = VISCOSITY * dt; // ν dt
        if (alpha <= 0.0f) return;

        // Copy RHS sources (original fields remain constant in the implicit solve)
        CopyArray(u, u0);
        CopyArray(v, v0);

        // Jacobi iterations for both components on their staggered grids
        for (int iter = 0; iter < DIFFUSION_ITERS; iter++)
        {
            // --- Jacobi for u (size: Nx+1 by Ny) ---
            for (int j = 0; j < GRID_HEIGHT; j++)
            {
                for (int i = 0; i <= GRID_WIDTH; i++)
                {
                    float sumN = 0.0f; int N = 0;
                    if (i > 0) { sumN += u[i - 1, j]; N++; }
                    if (i < GRID_WIDTH) { sumN += u[i + 1, j]; N++; }
                    if (j > 0) { sumN += u[i, j - 1]; N++; }
                    if (j < GRID_HEIGHT - 1) { sumN += u[i, j + 1]; N++; }

                    float denom = 1.0f + alpha * N;
                    uTmp[i, j] = (u0[i, j] + alpha * sumN) / denom;
                }
            }
            Swap(ref u, ref uTmp);

            // --- Jacobi for v (size: Nx by Ny+1) ---
            for (int j = 0; j <= GRID_HEIGHT; j++)
            {
                for (int i = 0; i < GRID_WIDTH; i++)
                {
                    float sumN = 0.0f; int N = 0;
                    if (i > 0) { sumN += v[i - 1, j]; N++; }
                    if (i < GRID_WIDTH - 1) { sumN += v[i + 1, j]; N++; }
                    if (j > 0) { sumN += v[i, j - 1]; N++; }
                    if (j < GRID_HEIGHT) { sumN += v[i, j + 1]; N++; }

                    float denom = 1.0f + alpha * N;
                    vTmp[i, j] = (v0[i, j] + alpha * sumN) / denom;
                }
            }
            Swap(ref v, ref vTmp);

            EnforceBoundaries();
        }
    }

    static void ComputeDivergenceAndRHS(float dt)
    {
        float scale = RHO / dt; // (rho/dt) * div
        for (int j = 0; j < GRID_HEIGHT; j++)
        {
            for (int i = 0; i < GRID_WIDTH; i++)
            {
                float dudx = u[i + 1, j] - u[i, j];
                float dvdy = v[i, j + 1] - v[i, j];
                float div = dudx + dvdy; // grid spacing h = 1
                divergence[i, j] = div;
                rhs[i, j] = scale * div;
            }
        }
    }

    // Solve pressure with Jacobi. Return last residual (L2 norm) for diagnostics.
    static float SolvePressureJacobi(int iterations)
    {
        float lastResidual = float.PositiveInfinity;
        for (int k = 0; k < iterations; k++)
        {
            float resAcc = 0.0f;
            for (int j = 0; j < GRID_HEIGHT; j++)
            {
                for (int i = 0; i < GRID_WIDTH; i++)
                {
                    float sumN = 0.0f; int N = 0;
                    if (i > 0) { sumN += pressure[i - 1, j]; N++; }
                    if (i < GRID_WIDTH - 1) { sumN += pressure[i + 1, j]; N++; }
                    if (j > 0) { sumN += pressure[i, j - 1]; N++; }
                    if (j < GRID_HEIGHT - 1) { sumN += pressure[i, j + 1]; N++; }

                    pressureTmp[i, j] = (N > 0) ? (sumN - rhs[i, j]) / N : 0.0f; // h=1
                }
            }
            Swap(ref pressure, ref pressureTmp);

            // compute residual every 8 iterations to save cost
            if ((k & 7) == 0 || k == iterations - 1)
            {
                resAcc = 0.0f;
                for (int j = 0; j < GRID_HEIGHT; j++)
                {
                    for (int i = 0; i < GRID_WIDTH; i++)
                    {
                        float lap = 0.0f;
                        int N = 0;
                        if (i > 0) { lap += pressure[i - 1, j]; N++; }
                        if (i < GRID_WIDTH - 1) { lap += pressure[i + 1, j]; N++; }
                        if (j > 0) { lap += pressure[i, j - 1]; N++; }
                        if (j < GRID_HEIGHT - 1) { lap += pressure[i, j + 1]; N++; }
                        float Ap = (N > 0) ? (lap - N * pressure[i, j]) : 0.0f;
                        float r = rhs[i, j] - Ap;
                        resAcc += r * r;
                    }
                }
                lastResidual = MathF.Sqrt(resAcc);
            }
        }
        return lastResidual;
    }

    static void ProjectVelocities(float dt)
    {
        float scale = dt / RHO; // (dt/rho) * grad p

        // u faces: i in [1..Nx-1], j in [0..Ny-1]
        for (int j = 0; j < GRID_HEIGHT; j++)
        {
            for (int i = 1; i < GRID_WIDTH; i++)
            {
                float gradp = pressure[i, j] - pressure[i - 1, j];
                u[i, j] -= scale * gradp;
            }
        }

        // v faces: i in [0..Nx-1], j in [1..Ny-1]
        for (int j = 1; j < GRID_HEIGHT; j++)
        {
            for (int i = 0; i < GRID_WIDTH; i++)
            {
                float gradp = pressure[i, j] - pressure[i, j - 1];
                v[i, j] -= scale * gradp;
            }
        }
    }

    // compute divergence after projection into divergencePost
    static void ComputeDivergencePost()
    {
        for (int j = 0; j < GRID_HEIGHT; j++)
        {
            for (int i = 0; i < GRID_WIDTH; i++)
            {
                float dudx = u[i + 1, j] - u[i, j];
                float dvdy = v[i, j + 1] - v[i, j];
                divergencePost[i, j] = dudx + dvdy;
            }
        }
    }

    // ===================== Helpers =====================

    static void CopyArray(float[,] src, float[,] dst)
    {
        int NX = src.GetLength(0), NY = src.GetLength(1);
        for (int j = 0; j < NY; j++)
            for (int i = 0; i < NX; i++)
                dst[i, j] = src[i, j];
    }

    // ===================== Optional: simple velocity injection near cursor =====================

    static void InjectVelocity(Vector2 gridPos, Vector2 impulse)
    {
        // Add to nearby u and v faces with bilinear weights around the cursor cell
        float xu = gridPos.X;
        float yu = gridPos.Y - 0.5f;
        AddBilinear(u, xu, yu, impulse.X);
        AddBilinear(u, xu - 1, yu, impulse.X);
        AddBilinear(u, xu + 1, yu, impulse.X);
        AddBilinear(u, xu, yu + 1, impulse.X);
        AddBilinear(u, xu, yu - 1, impulse.X);

        float xv = gridPos.X - 0.5f;
        float yv = gridPos.Y;
        AddBilinear(v, xv, yv, impulse.Y);
        AddBilinear(v, xv, yv - 1, impulse.Y);
        AddBilinear(v, xv, yv + 1, impulse.Y);
        AddBilinear(v, xv + 1, yv, impulse.Y);
        AddBilinear(v, xv - 1, yv, impulse.Y);
    }

    static void AddBilinear(float[,] A, float x, float y, float value)
    {
        int NX = A.GetLength(0);
        int NY = A.GetLength(1);

        const float EPS = 1e-4f;
        float xc = MathF.Max(0.0f, MathF.Min(x, NX - 1 - EPS));
        float yc = MathF.Max(0.0f, MathF.Min(y, NY - 1 - EPS));

        int i0 = (int)MathF.Floor(xc);
        int j0 = (int)MathF.Floor(yc);
        int i1 = Math.Min(i0 + 1, NX - 1);
        int j1 = Math.Min(j0 + 1, NY - 1);

        float sx = xc - i0;
        float sy = yc - j0;

        float w00 = (1 - sx) * (1 - sy);
        float w10 = sx * (1 - sy);
        float w01 = (1 - sx) * sy;
        float w11 = sx * sy;

        A[i0, j0] += value * w00;
        A[i1, j0] += value * w10;
        A[i0, j1] += value * w01;
        A[i1, j1] += value * w11;
    }

    // scan arrays for NaN/Inf
    static bool ScanForNaNInf()
    {
        for (int j = 0; j < GRID_HEIGHT; j++)
            for (int i = 0; i < GRID_WIDTH; i++)
                if (float.IsNaN(pressure[i, j]) || float.IsInfinity(pressure[i, j]) ||
                    float.IsNaN(density[i, j]) || float.IsInfinity(density[i, j]))
                    return true;

        for (int j = 0; j < GRID_HEIGHT; j++)
            for (int i = 0; i <= GRID_WIDTH; i++)
                if (float.IsNaN(u[i, j]) || float.IsInfinity(u[i, j])) return true;

        for (int j = 0; j <= GRID_HEIGHT; j++)
            for (int i = 0; i < GRID_WIDTH; i++)
                if (float.IsNaN(v[i, j]) || float.IsInfinity(v[i, j])) return true;

        return false;
    }

    // ===================== Rendering =====================

    static void Draw()
    {
        Raylib.BeginDrawing();
        Raylib.ClearBackground(Color.Black);

        // Draw density as grayscale tiles
        for (int y = 0; y < GRID_HEIGHT; y++)
        {
            for (int x = 0; x < GRID_WIDTH; x++)
            {
                byte c = (byte)(Math.Clamp(density[x, y], 0.0f, 1.0f) * 255);
                Raylib.DrawRectangle(
                    (int)(x * CELL_SIZE),
                    (int)(y * CELL_SIZE),
                    (int)CELL_SIZE,
                    (int)CELL_SIZE,
                    new Color(c, c, c, (byte)255)
                );
            }
        }

        // Optional: overlay velocity vectors (press V to toggle later if desired)
        DrawVelocityField(6);

        // Diagnostics overlay
        diag.DrawOverlay();

        // small divergence preview (right side)
        DrawDivergencePreview();

        Raylib.EndDrawing();
    }

    static void DrawVelocityField(int stride)
    {
        // Draws sparse velocity glyphs to sanity-check directions
        for (int j = 0; j < GRID_HEIGHT; j += stride)
        {
            for (int i = 0; i < GRID_WIDTH; i += stride)
            {
                Vector2 pos = new Vector2((i + 0.5f) * CELL_SIZE, (j + 0.5f) * CELL_SIZE);
                Vector2 vel = SampleVelocity(new Vector2(i + 0.5f, j + 0.5f));
                Vector2 tip = pos + vel * CELL_SIZE * 0.2f; // scale for display
                Raylib.DrawLine((int)pos.X, (int)pos.Y, (int)tip.X, (int)tip.Y, Color.DarkGreen);
            }
        }
    }

    static void DrawDivergencePreview()
    {
        int previewSize = 128;
        int px = (int)(GRID_WIDTH * CELL_SIZE) - previewSize - 10;
        int py = 10;
        int cell = Math.Max(1, previewSize / GRID_WIDTH);

        // Draw small background
        Raylib.DrawRectangle(px - 2, py - 2, previewSize + 4, previewSize + 4, Color.DarkGray);

        // Draw divergencePost scaled into preview area (simple downsample)
        for (int pyi = 0; pyi < previewSize; pyi += cell)
        {
            for (int pxi = 0; pxi < previewSize; pxi += cell)
            {
                // map preview pixel to grid cell
                int gi = (int)((float)pxi / previewSize * GRID_WIDTH);
                int gj = (int)((float)pyi / previewSize * GRID_HEIGHT);
                if (gi < 0) gi = 0; if (gi >= GRID_WIDTH) gi = GRID_WIDTH - 1;
                if (gj < 0) gj = 0; if (gj >= GRID_HEIGHT) gj = GRID_HEIGHT - 1;
                float d = divergencePost[gi, gj];
                // color ramp: negative blue, zero black, positive red
                float s = MathF.Tanh(MathF.Abs(d) * 10.0f); // clamp & scale
                Color col = d > 0 ? ColorLerp(Color.Black, Color.Red, s) : ColorLerp(Color.Black, Color.SkyBlue, s);
                Raylib.DrawRectangle(px + pxi, py + pyi, cell, cell, col);
            }
        }
    }

    static Color ColorLerp(Color a, Color b, float t)
    {
        t = MathF.Max(0, MathF.Min(1, t));
        return new Color(
            (byte)(a.R + (b.R - a.R) * t),
            (byte)(a.G + (b.G - a.G) * t),
            (byte)(a.B + (b.B - a.B) * t),
            (byte) 255
        );
    }

    // ===================== Diagnostics class =====================

    class Diagnostics
    {
        // ring buffer length
        const int LEN = 300;
        int idx = 0;
        int filled = 0;

        // time series
        float[] massHist = new float[LEN];
        float[] tvHist = new float[LEN];
        float[] keHist = new float[LEN];
        float[] divPreMaxHist = new float[LEN];
        float[] divPostMaxHist = new float[LEN];
        float[] presResHist = new float[LEN];
        float[] cflHist = new float[LEN];
        float[] fpsHist = new float[LEN];

        // last frame quick stats
        public float lastMass = 0;
        public float lastTV = 0;
        public float lastKE = 0;
        public float lastMaxSpeed = 0;
        public float lastDivPreMax = 0;
        public float lastDivPostMax = 0;
        public float lastPressureResidual = 0;
        public float lastCFL = 0;
        public int frames = 0;

        // display control
        bool show = true;

        // font/placement
        int left = 8;
        int top = 8;

        public void toggleShow() { show = !show; }

        public void ClearHistory()
        {
            idx = 0; filled = 0;
            Array.Clear(massHist, 0, LEN);
            Array.Clear(tvHist, 0, LEN);
            Array.Clear(keHist, 0, LEN);
            Array.Clear(divPreMaxHist, 0, LEN);
            Array.Clear(divPostMaxHist, 0, LEN);
            Array.Clear(presResHist, 0, LEN);
            Array.Clear(cflHist, 0, LEN);
            Array.Clear(fpsHist, 0, LEN);
        }

        public void GatherAll(
            float[,] density,
            float[,] u, float[,] v,
            float[,] divPre, float[,] divPost,
            float[,] pressure,
            float pressureResidual,
            float dt,
            long msAdvection,
            long msForces,
            long msDiffuse,
            long msDivergence,
            long msPressure,
            long msProject,
            long msBoundaries,
            long msFrame)
        {
            // mass & TV
            float mass = 0f;
            float tv = 0f;
            for (int j = 0; j < GRID_HEIGHT; j++)
            {
                for (int i = 0; i < GRID_WIDTH; i++)
                {
                    float d = density[i, j];
                    mass += d;
                    if (i + 1 < GRID_WIDTH) tv += MathF.Abs(density[i + 1, j] - d);
                    if (j + 1 < GRID_HEIGHT) tv += MathF.Abs(density[i, j + 1] - d);
                }
            }

            // kinetic energy & max speed
            float ke = 0f;
            float maxSpeed = 0f;
            for (int j = 0; j < GRID_HEIGHT; j++)
            {
                for (int i = 0; i <= GRID_WIDTH; i++)
                {
                    float val = u[i, j];
                    ke += 0.5f * val * val;
                    if (MathF.Abs(val) > maxSpeed) maxSpeed = MathF.Abs(val);
                }
            }
            for (int j = 0; j <= GRID_HEIGHT; j++)
            {
                for (int i = 0; i < GRID_WIDTH; i++)
                {
                    float val = v[i, j];
                    ke += 0.5f * val * val;
                    if (MathF.Abs(val) > maxSpeed) maxSpeed = MathF.Abs(val);
                }
            }

            // divergence norms
            float divPreMax = 0f;
            float divPreL2 = 0f;
            float divPostMax = 0f;
            float divPostL2 = 0f;
            for (int j = 0; j < GRID_HEIGHT; j++)
            {
                for (int i = 0; i < GRID_WIDTH; i++)
                {
                    float a = divPre[i, j];
                    float b = divPost[i, j];
                    divPreMax = MathF.Max(divPreMax, MathF.Abs(a));
                    divPostMax = MathF.Max(divPostMax, MathF.Abs(b));
                    divPreL2 += a * a;
                    divPostL2 += b * b;
                }
            }
            divPreL2 = MathF.Sqrt(divPreL2);
            divPostL2 = MathF.Sqrt(divPostL2);

            // CFL
            float cfl = maxSpeed * dt; // h = 1

            // store
            lastMass = mass;
            lastTV = tv;
            lastKE = ke;
            lastMaxSpeed = maxSpeed;
            lastDivPreMax = divPreMax;
            lastDivPostMax = divPostMax;
            lastPressureResidual = pressureResidual;
            lastCFL = cfl;

            massHist[idx] = mass;
            tvHist[idx] = tv;
            keHist[idx] = ke;
            divPreMaxHist[idx] = divPreMax;
            divPostMaxHist[idx] = divPostMax;
            presResHist[idx] = pressureResidual;
            cflHist[idx] = cfl;
            fpsHist[idx] = Raylib.GetFPS();

            idx = (idx + 1) % LEN;
            filled = Math.Min(filled + 1, LEN);
            frames++;
        }

        // Draw overlay UI (compact)
        public void DrawOverlay()
        {
            if (!show) return;

            int x = left;
            int y = top;
            int lineh = 16;

            // background rectangle
            Raylib.DrawRectangle(x - 6, y - 6, 420, 220, new Color(0, 0, 0, 150));

            // header
            Raylib.DrawText("Diagnostics (D toggles) — P: pause, N: step, C: clear history", x, y, 12, Color.LightGray);
            y += lineh;

            Raylib.DrawText($"Frame: {frames}", x, y, 12, Color.LightGray);
            y += lineh;

            Raylib.DrawText($"Mass: {lastMass:F3}   TV: {lastTV:F3}   KE: {lastKE:F3}", x, y, 12, Color.LightGray);
            y += lineh;

            Raylib.DrawText($"MaxSpeed: {lastMaxSpeed:F3}   CFL:{lastCFL:F3}   PresRes:{lastPressureResidual:E3}", x, y, 12, Color.LightGray);
            y += lineh;

            Raylib.DrawText($"DivPreMax: {lastDivPreMax:E3}   DivPostMax: {lastDivPostMax:E3}", x, y, 12, Color.LightGray);
            y += lineh;

            // simple small sparklines (mass, KE, divPost)
            DrawSparkline("mass", massHist, x, y, 200, 40, filled, idx); y += 44;
            DrawSparkline("KE", keHist, x, y, 200, 40, filled, idx); y += 44;
            DrawSparkline("divPostMax", divPostMaxHist, x, y, 200, 40, filled, idx); y += 44;
        }

        void DrawSparkline(string label, float[] buffer, int x, int y, int w, int h, int filled, int idx)
        {
            Raylib.DrawText(label, x, y, 10, Color.LightGray);
            int bx = x + 60;
            int by = y;
            Raylib.DrawRectangle(bx - 2, by - 2, w + 4, h + 4, new Color(80, 80, 80, 120));
            if (filled == 0) return;
            // find range
            float min = float.MaxValue, max = float.MinValue;
            for (int i = 0; i < filled; i++)
            {
                float v = buffer[i];
                if (float.IsNaN(v) || float.IsInfinity(v)) continue;
                if (v < min) min = v;
                if (v > max) max = v;
            }
            if (min == float.MaxValue) min = 0; // avoid
            if (min == max) max = min + 1e-6f;
            // draw polyline
            int display = Math.Min(filled, w);
            for (int i = 0; i < display - 1; i++)
            {
                int a = (idx - i - 1 + LEN) % LEN;
                int b = (idx - i - 2 + LEN) % LEN;
                float va = buffer[a];
                float vb = buffer[b];
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
