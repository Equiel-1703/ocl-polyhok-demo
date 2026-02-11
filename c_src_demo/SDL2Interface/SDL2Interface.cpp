#include "SDL2Interface.hpp"

SDL2Interface::SDL2Interface()
{
  if (SDL_Init(SDL_INIT_VIDEO) < 0)
  {
    std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
    throw std::runtime_error("Failed to initialize SDL");
  }
}

SDL2Interface::~SDL2Interface()
{
  if (texture)
  {
    SDL_DestroyTexture(texture);
  }
  if (renderer)
  {
    SDL_DestroyRenderer(renderer);
  }
  if (window)
  {
    SDL_DestroyWindow(window);
  }

  SDL_Quit();
}

void SDL2Interface::createWindow(const char *title, int width, int height)
{
  std::cout << "[SDL2 Interface] Creating window with title: '" << title << "'." << std::endl;

  windowWidth = width;
  windowHeight = height;

  window = SDL_CreateWindow(
      title,
      SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
      windowWidth, windowHeight,
      SDL_WINDOW_SHOWN);

  if (!window)
  {
    std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
    throw std::runtime_error("Failed to create SDL window");
  }

  renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
  if (!renderer)
  {
    std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
    throw std::runtime_error("Failed to create SDL renderer");
  }

  texture = SDL_CreateTexture(
      renderer,
      SDL_PIXELFORMAT_ARGB8888,    // 32-bit color (check other types if needed) 0xAARRGGBB
      SDL_TEXTUREACCESS_STREAMING, // Optimized for frequent updates
      width, height);

  if (!texture)
  {
    std::cerr << "Texture could not be created! SDL_Error: " << SDL_GetError() << std::endl;
    throw std::runtime_error("Failed to create SDL texture");
  }

  pixels.resize(windowWidth * windowHeight, 0); // Initialize pixel buffer
}

void SDL2Interface::updateTexture(const std::vector<uint32_t> &newPixels)
{
  if (newPixels.size() != pixels.size())
  {
    std::cerr << "Pixel data size mismatch!" << std::endl;
    return;
  }

  pixels = newPixels;

  // Update the texture with the new pixel data
  SDL_UpdateTexture(texture, nullptr, pixels.data(), static_cast<int>(windowWidth * sizeof(uint32_t)));
}

void SDL2Interface::render()
{
  // Clear screen
  SDL_RenderClear(renderer);
  // Copy texture to renderer
  SDL_RenderCopy(renderer, texture, nullptr, nullptr);
  // Update screen
  SDL_RenderPresent(renderer);
}

void SDL2Interface::handleEvents(bool &quit)
{
  while (SDL_PollEvent(&e) != 0)
  {
    if (e.type == SDL_QUIT)
      quit = true;
    if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)
      quit = true;
  }
}