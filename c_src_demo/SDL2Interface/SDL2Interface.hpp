#pragma once

#include <SDL2/SDL.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <stdexcept>

class SDL2Interface
{
private:
  int windowWidth = 0;
  int windowHeight = 0;

  SDL_Window *window = nullptr;
  SDL_Renderer *renderer = nullptr;
  SDL_Texture *texture = nullptr;
  SDL_Event e;

  std::vector<uint32_t> pixels;

public:
  SDL2Interface();
  ~SDL2Interface();

  void createWindow(const char *title, int width, int height);
  void updateTexture(const std::vector<uint32_t> &newPixels);
  void render();
  void handleEvents(bool &quit);
};
