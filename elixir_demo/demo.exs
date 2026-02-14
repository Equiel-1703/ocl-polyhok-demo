require OCLPolyHok

defmodule SDL2 do
  @on_load :load_nifs
  def load_nifs do
    :erlang.load_nif(~c"priv/demo_nifs", 0)
  end

  def loop() do
    receive do
      {:render_img, nx_array} ->
        update_image_nx(nx_array)

      msg ->
        IO.inspect(msg, label: "SDL2 received an unknown message: ")
    end

    # Verifica se a janela do SDL2 recebeu um pedido de fechamento (como clicar no "X" da janela)
    case close_requested_nif() do
      true ->
        IO.puts("SDL process ending. Close requested.")
        :ok

      false ->
        # Se não houve pedido de encerramento, chama a NIF para renderizar a janela e continua o loop
        render_window_nif()
        loop()
    end
  end

  # Recebe um Nx armazenado na RAM e envia para a NIF de atualização da textura do SDL2
  defp update_image_nx(%Nx.Tensor{data: %Nx.BinaryBackend{state: array}}) do
    update_image_nif(array)
  end

  # --- NIF stubs ---
  def create_window_nif(_title, _width, _height) do
    :erlang.nif_error(:nif_not_implemented)
  end

  def close_requested_nif() do
    :erlang.nif_error(:nif_not_implemented)
  end

  def render_window_nif() do
    :erlang.nif_error(:nif_not_implemented)
  end

  def update_image_nif(_img_array) do
    :erlang.nif_error(:nif_not_implemented)
  end
end

OCLPolyHok.defmodule RayTracer do
  defk render_kernel(img_array, dim_x, dim_y) do
    tid_x = get_global_id(0)
    tid_y = get_global_id(1)

    if tid_x < dim_x && tid_y < dim_y do
      # Blue color in RGB888 format (0xRRGGBB)
      img_array[tid_x * dim_x + tid_y] = 0x0000FF
    end
  end

  defp render(sdl_pid, monitor_ref, gpu_array, dim_x, dim_y) do
    receive do
      {:DOWN, ^monitor_ref, :process, _pid, _reason} ->
        # Se o processo do SDL for encerrado, saímos do loop de renderização lindamente =)
        IO.puts("SDL process has terminated. Stopping render loop.")
        :ok

      msg ->
        IO.inspect(msg, label: "Unknown message :")
        :ok
    after
      0 ->
        threads_per_block = 16
        blocks_x = div(dim_x + threads_per_block - 1, threads_per_block)
        blocks_y = div(dim_y + threads_per_block - 1, threads_per_block)

        # Lançando o kernel de renderização na GPU
        OCLPolyHok.spawn(
          &RayTracer.render_kernel/3,
          {blocks_x, blocks_y, 1},
          {threads_per_block, threads_per_block, 1},
          [gpu_array, dim_x, dim_y]
        )

        # Pega o resultado da renderização na GPU e envia para o processo do SDL2 para atualizar a textura
        result_ram = gpu_array |> OCLPolyHok.get_gnx()
        send(sdl_pid, {:render_img, result_ram})

        # Recursão para continuar o loop de renderização
        render(sdl_pid, monitor_ref, gpu_array, dim_x, dim_y)
    end
  end

  def render_loop(sdl_pid, gpu_array, dim_x, dim_y) do
    monitor_ref = Process.monitor(sdl_pid)

    render(sdl_pid, monitor_ref, gpu_array, dim_x, dim_y)
  end
end

# Estrutura do programa:
# * um processo vai cuidar da janela -> SDL2
# * outro processo (o principal) da renderização -> RayTracer

window_width = 500
window_height = 500

# Array com a imagem na GPU - global
img_array_gpu = OCLPolyHok.new_gnx(window_width, window_height, {:s, 32})

# Lançando o processo do SDL2
SDL2.create_window_nif(~c"teste", window_width, window_height)
sdl_process_pid = spawn(fn -> SDL2.loop() end)

# Processo principal cuida da renderização, e envia a imagem renderizada para o processo do SDL2
RayTracer.render_loop(sdl_process_pid, img_array_gpu, window_width, window_height)
