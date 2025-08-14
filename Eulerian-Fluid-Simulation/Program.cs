using Raylib_cs;
using System;
using System.Numerics;

public class Program
{
    // Grid dimensions
    const int GRID_WIDTH = 64;
    const int GRID_HEIGHT = 64;
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

    // Mouse state for simple velocity injection
    static Vector2 prevMouse;
    const float MOUSE_FORCE_SCALE = 6.0f; // tweakable impulse strength

    public static void Main()
    {
        int screenWidth = (int)(GRID_WIDTH * CELL_SIZE);
        int screenHeight = (int)(GRID_HEIGHT * CELL_SIZE);

        Raylib.InitWindow(screenWidth, screenHeight, "2D Eulerian Fluid Simulation - MAC Grid (Advection)");
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

        // --- Advection step ---
        AdvectDensity(dt);
        AdvectVelocity(dt);

        // (No forces, diffusion, or projection yet.)
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
        // Reconstruct continuous velocity from staggered components
        float ux = SampleU(u, pos);
        float vy = SampleV(v, pos);
        return new Vector2(ux, vy);
    }

    static float SampleScalar(float[,] A, Vector2 pos)
    {
        // Scalars live at cell centers (i+0.5, j+0.5)
        float xs = pos.X - 0.5f; // convert to index space
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
        // Clamp to valid interpolation domain so i1/j1 exist
        const float EPS = 1e-4f;
        float xc = MathF.Max(0.0f, MathF.Min(x, NX - 1 - EPS));
        float yc = MathF.Max(0.0f, MathF.Min(y, NY - 1 - EPS));

        int i0 = (int)MathF.Floor(xc);
        int j0 = (int)MathF.Floor(yc);
        int i1 = i0 + 1;
        int j1 = j0 + 1;

        float sx = xc - i0;
        float sy = yc - j0;

        // Guard edges (when NX==1 or NY==1) — keep inside bounds
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
        // Scalars live at centers in [0.5, Nx-0.5] × [0.5, Ny-0.5]
        float x = MathF.Max(0.5f, MathF.Min(p.X, GRID_WIDTH - 0.5f));
        float y = MathF.Max(0.5f, MathF.Min(p.Y, GRID_HEIGHT - 0.5f));
        return new Vector2(x, y);
    }

    static Vector2 ClampToUDomain(Vector2 p)
    {
        // U faces live at x∈[0..Nx], y∈[0.5..Ny-0.5]
        float x = MathF.Max(0.0f, MathF.Min(p.X, GRID_WIDTH));
        float y = MathF.Max(0.5f, MathF.Min(p.Y, GRID_HEIGHT - 0.5f));
        return new Vector2(x, y);
    }

    static Vector2 ClampToVDomain(Vector2 p)
    {
        // V faces live at x∈[0.5..Nx-0.5], y∈[0..Ny]
        float x = MathF.Max(0.5f, MathF.Min(p.X, GRID_WIDTH - 0.5f));
        float y = MathF.Max(0.0f, MathF.Min(p.Y, GRID_HEIGHT));
        return new Vector2(x, y);
    }

    static void Swap(ref float[,] A, ref float[,] B)
    {
        var tmp = A; A = B; B = tmp;
    }

    // ===================== Optional: simple velocity injection near cursor =====================

    static void InjectVelocity(Vector2 gridPos, Vector2 impulse)
    {
        // Add to nearby u and v faces with bilinear weights around the cursor cell
        // This is purely for interactive testing to see density move.

        // Affect u faces around (i0..i0+1, j0..j0+1) in u-index space
        // Map gridPos to u index space: (x_u = x, y_u = y-0.5)
        float xu = gridPos.X;
        float yu = gridPos.Y - 0.5f;
        AddBilinear(u, xu, yu, impulse.X);

        // Affect v faces around (i0..i0+1, j0..j0+1) in v-index space
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