using Raylib_cs;
using System;
using System.Numerics;

public class Program
{
    // Grid dimensions
    const int GRID_WIDTH = 100;
    const int GRID_HEIGHT = 100;
    const float CELL_SIZE = 8.0f; // Pixels per cell (rendering only)

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
    const float GRAVITY_Y = -3.0f;       // body force y (down)
    const float VISCOSITY = 0.0005f;     // kinematic viscosity (set 0 to disable)
    const int DIFFUSION_ITERS = 30;    // Jacobi iterations for diffusion
    const int PRESSURE_ITERS = 200;   // Jacobi iterations for Poisson

    // Poisson / divergence buffers (cell-centered)
    static float[,] pressureTmp = new float[GRID_WIDTH, GRID_HEIGHT];
    static float[,] rhs = new float[GRID_WIDTH, GRID_HEIGHT];
    static float[,] divergence = new float[GRID_WIDTH, GRID_HEIGHT];

    // Diffusion RHS copies (staggered)
    static float[,] u0 = new float[GRID_WIDTH + 1, GRID_HEIGHT];
    static float[,] v0 = new float[GRID_WIDTH, GRID_HEIGHT + 1];

    // Mouse state for simple velocity injection
    static Vector2 prevMouse;
    const float MOUSE_FORCE_SCALE = 15.0f; // tweakable impulse strength

    public static void Main()
    {
        int screenWidth = (int)(GRID_WIDTH * CELL_SIZE);
        int screenHeight = (int)(GRID_HEIGHT * CELL_SIZE);

        Raylib.InitWindow(screenWidth, screenHeight, "2D Eulerian Fluid Simulation - MAC Grid (Advection+Projection)");
        Raylib.SetTargetFPS(120);
        prevMouse = Raylib.GetMousePosition();

        while (!Raylib.WindowShouldClose())
        {
            Update();
            Draw();
        }

        Raylib.CloseWindow();
    }

    static void Update()
    {
        float dt = DT; // fixed for now

        // Density injection with left mouse button
        if (Raylib.IsMouseButtonDown(MouseButton.Left))
        {
            Vector2 mouse = Raylib.GetMousePosition();
            int gx = (int)(mouse.X / CELL_SIZE);
            int gy = (int)(mouse.Y / CELL_SIZE);

            if (gx >= 0 && gx < GRID_WIDTH && gy >= 0 && gy < GRID_HEIGHT)
            {
                density[gx, gy] = MathF.Min(1.0f, density[gx, gy] + 0.2f);
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

        // --- Core fluid step order ---
        // 1) Advect scalars and velocity by current velocity
        AdvectDensity(dt);
        AdvectVelocity(dt);

        // 2) Apply body forces (e.g., gravity)
        ApplyForces(dt);

        // 3) Implicit diffusion (viscosity)
        if (VISCOSITY > 0.0f)
            DiffuseVelocities(dt);

        // 4) Enforce simple box boundary before projection
        EnforceBoundaries();

        // 5) Build divergence and RHS for pressure Poisson
        ComputeDivergenceAndRHS(dt);

        // 6) Solve for pressure (Jacobi)
        SolvePressureJacobi(PRESSURE_ITERS);

        // 7) Subtract pressure gradient (projection)
        ProjectVelocities(dt);

        // 8) Final boundary clamp
        EnforceBoundaries();
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

    static void SolvePressureJacobi(int iterations)
    {
        // Start from previous pressure; Jacobi will relax toward the solution.
        for (int k = 0; k < iterations; k++)
        {
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
        }
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

        float xv = gridPos.X - 0.5f;
        float yv = gridPos.Y;
        AddBilinear(v, xv, yv, impulse.Y);
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
                    new Color(c, c, c, (byte) 255)
                );
            }
        }

        // Optional: overlay velocity vectors (press V to toggle later if desired)
        DrawVelocityField(12);

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
}
