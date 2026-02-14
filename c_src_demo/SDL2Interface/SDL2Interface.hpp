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

public:
  SDL2Interface();
  ~SDL2Interface();

  void createWindow(const char *title, int width, int height);

  // Estou usando int32_t para os pixels porque o formato da textura é RGB888,
  // que é um formato de 24 bits (8 bits para cada canal de cor e ignora o canal alpha).
  // Isso é perfeito pois em Elixir o Nx usa inteiros de 32 bits com sinal, 
  // e o byte mais significativo (MSB) pode ser ignorado, já que o formato RGB888 não usa o canal alpha.
  void updateTexture(int32_t *newPixels);
  size_t getTextureSizeBytes() const { return sizeof(int32_t) * windowWidth * windowHeight; }
  
  void render();
  void handleEvents(bool &quit);

  int getWindowWidth() const { return windowWidth; }
  int getWindowHeight() const { return windowHeight; }
};
