#pragma once

#include <SDL2/SDL.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <atomic>

class SDL2Interface
{
private:
  // Well, I just discovered -- IN THE FUCKING WORST WAY POSSIBLE -- that SDL2 is NOT thread safe.
  // So, I'll have to make sure that all SDL2 calls are made from the same thread. The Erlang BEAM VM can
  // schedules different threads for NIF calls, so this is breaking the GUI and making the system unstable.

  // SDL2 thread
  std::thread sdlThread;
  
  // Pixel buffer (shared between threads, protected by mutex)
  std::vector<int32_t> pixelBuffer;
  std::mutex bufferMutex;

  // Control flags
  std::atomic<bool> quit;
  std::atomic<bool> hasNewPixels;

  // Window stuff
  int windowWidth = 0;
  int windowHeight = 0;
  std::string windowTitle;

  // SDL2 objects (ONLY ACCESSED FROM THE SDL2 THREAD)
  SDL_Window *window = nullptr;
  SDL_Renderer *renderer = nullptr;
  SDL_Texture *texture = nullptr;

  // This is the main loop that runs in the SDL2 thread, responsible for handling events and rendering.
  void sdlMainLoop();

public:
  SDL2Interface();
  ~SDL2Interface();

  void createWindow(const char *title, int width, int height);

  // Estou usando int32_t para os pixels porque o formato da textura é RGB888,
  // que é um formato de 24 bits (8 bits para cada canal de cor e ignora o canal alpha).
  // Isso é perfeito pois em Elixir o Nx usa inteiros de 32 bits com sinal, 
  // e o byte mais significativo (MSB) pode ser ignorado, já que o formato RGB888 não usa o canal alpha.
  // E agora esse CARALHO é thread safe, pois o buffer de pixels é protegido por um mutex e as flags de controle são atômicas.
  void updateTexture(int32_t *newPixels);

  // Thread-safe check if the window has been requested to close
  bool isCloseRequested() const { return quit.load(); }

  size_t getTextureSizeBytes() const { return sizeof(int32_t) * windowWidth * windowHeight; }
  int getWindowWidth() const { return windowWidth; }
  int getWindowHeight() const { return windowHeight; }
};
