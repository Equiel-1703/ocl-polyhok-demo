#include "SDL2Interface.hpp"

SDL2Interface::SDL2Interface()
{
  // Initialize atomic flags
  quit = false;
  hasNewPixels = false;
}

SDL2Interface::~SDL2Interface()
{
  // Signal the SDL thread to quit and wait for it to finish
  quit = true;

  if (sdlThread.joinable())
  {
    // Wait for the SDL thread to finish
    sdlThread.join();
  }
}

void SDL2Interface::createWindow(const char *title, int width, int height)
{
  this->windowTitle.assign(title);
  this->windowWidth = width;
  this->windowHeight = height;

  // Allocate pixel buffer with the correct size
  pixelBuffer.resize(windowWidth * windowHeight);

  // Start the SDL thread
  this->sdlThread = std::thread(&SDL2Interface::sdlMainLoop, this);
}

void SDL2Interface::sdlMainLoop()
{
  // Initialize SDL (MUST be in the same thread that all other SDL calls)
  if (SDL_Init(SDL_INIT_VIDEO) < 0)
  {
    std::cerr << "[C++ SDL2Interface] Failed to initialize SDL: " << SDL_GetError() << std::endl;
    return;
  }

  // Create window
  this->window = SDL_CreateWindow(
      windowTitle.c_str(),
      SDL_WINDOWPOS_CENTERED,
      SDL_WINDOWPOS_CENTERED,
      windowWidth, windowHeight,
      SDL_WINDOW_SHOWN);

  if (!window)
  {
    std::cerr << "[C++ SDL2Interface] Failed to create window: " << SDL_GetError() << std::endl;
    SDL_Quit();
    return;
  }

  // Create renderer
  this->renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

  if (!renderer)
  {
    std::cerr << "[C++ SDL2Interface] Failed to create renderer: " << SDL_GetError() << std::endl;
    SDL_DestroyWindow(window);
    SDL_Quit();
    return;
  }

  // Create texture -- this is what we will be updating with new pixel data
  this->texture = SDL_CreateTexture(
      renderer,
      SDL_PIXELFORMAT_RGB888,      // 24 bits per pixel (8 bits for each color channel, no alpha); We can use normal int32_t for the pixel buffer since the MSB will be ignored.
      SDL_TEXTUREACCESS_STREAMING, // We will be updating the texture frequently with new pixel data. This optimizes for that use case.
      windowWidth, windowHeight);

  if (!texture)
  {
    std::cerr << "[C++ SDL2Interface] Failed to create texture: " << SDL_GetError() << std::endl;
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return;
  }

  // Main loop
  while (!quit.load())
  {
    // Handle events
    SDL_Event e;
    while (SDL_PollEvent(&e) != 0)
    {
      // Check if the user has requested to close the window (e.g., by clicking the close button)
      if (e.type == SDL_QUIT)
      {
        quit = true;
      }

      // Check if ESC key is pressed. We can close the window with ESC key as well.
      if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)
      {
        quit = true;
      }
    }

    // If we have new pixel data, update the texture
    if (hasNewPixels.load())
    {
      // Lock the buffer mutex to safely access the pixel buffer
      std::lock_guard<std::mutex> lock(bufferMutex);
      SDL_UpdateTexture(texture, nullptr, pixelBuffer.data(), windowWidth * sizeof(int32_t));
      hasNewPixels = false; // Reset the flag after updating the texture
    }

    // Render screen
    SDL_RenderClear(renderer);                           // Clear the renderer with the current draw color (default is black)
    SDL_RenderCopy(renderer, texture, nullptr, nullptr); // Copy the entire texture to the renderer
    SDL_RenderPresent(renderer);                         // Update the screen with the rendered content

    // Sleep for a short duration to limit the frame rate and reduce CPU usage
    std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
  }

  std::cerr << "[C++ SDL2Interface] SDL main loop has exited and resources have been cleaned up." << std::endl;

  // When we exit the main loop, clean up SDL resources
  SDL_DestroyTexture(texture);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
}

// Thread-safe method to update the texture with new pixel data from the main thread (called by NIFs)
void SDL2Interface::updateTexture(int32_t *newPixels)
{
  // Lock the buffer mutex
  std::lock_guard<std::mutex> lock(bufferMutex);
  // Update the pixel buffer with the new pixel data
  std::copy(newPixels, newPixels + (windowWidth * windowHeight), pixelBuffer.begin());
  // Set the flag to indicate that we have new pixels to update the texture with
  hasNewPixels = true;
}