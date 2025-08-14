using Raylib_cs;
using System;
using System.Numerics;

public class Program
{
    // Grid dimensions
    const int GRID_WIDTH = 64;
    const int GRID_HEIGHT = 64;
    const float CELL_SIZE = 8.0f; // Pixels per cell

    // Time step
    const float DT = 0.1f;

    // Fluid properties
    static float[,] pressure = new float[GRID_WIDTH, GRID_HEIGHT];
    static float[,] density = new float[GRID_WIDTH, GRID_HEIGHT];

    // MAC grid velocities:
    // u: horizontal velocity at vertical cell edges (GRID_WIDTH+1 by GRID_HEIGHT)
    // v: vertical velocity at horizontal cell edges (GRID_WIDTH by GRID_HEIGHT+1)
    static float[,] u = new float[GRID_WIDTH + 1, GRID_HEIGHT];
    static float[,] v = new float[GRID_WIDTH, GRID_HEIGHT + 1];

    public static void Main()
    {
        int screenWidth = (int)(GRID_WIDTH * CELL_SIZE);
        int screenHeight = (int)(GRID_HEIGHT * CELL_SIZE);

        Raylib.InitWindow(screenWidth, screenHeight, "2D Eulerian Fluid Simulation - MAC Grid");
        Raylib.SetTargetFPS(120);

        // Main loop
        while (!Raylib.WindowShouldClose())
        {
            Update();
            Draw();
        }

        Raylib.CloseWindow();
    }

    static void Update()
    {
        // This is where simulation steps will go
        // For now, let's just add density at mouse position
        if (Raylib.IsMouseButtonDown(MouseButton.Left))
        {
            Vector2 mouse = Raylib.GetMousePosition();
            int gx = (int)(mouse.X / CELL_SIZE);
            int gy = (int)(mouse.Y / CELL_SIZE);

            if (gx >= 0 && gx < GRID_WIDTH && gy >= 0 && gy < GRID_HEIGHT)
            {
                density[gx, gy] = density[gx, gy] > 1.0f ? 1.0f : density[gx, gy] + 0.1f;
            }
        }
    }

    static void Draw()
    {
        Raylib.BeginDrawing();
        Raylib.ClearBackground(Color.Black);

        // Draw density as grayscale
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

        Raylib.EndDrawing();
    }
}
