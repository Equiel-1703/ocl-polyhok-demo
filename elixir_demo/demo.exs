require OCLPolyHok

# OCLPolyHok.set_debug_logs(true)

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
        IO.inspect(msg, label: "[SDL2] Received an unknown message")
    end

    # Verifica se a janela do SDL2 recebeu um pedido de fechamento (como clicar no "X" da janela)
    case close_requested_nif() do
      true ->
        IO.puts("[SDL2] Close requested. Exiting process.")
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

OCLPolyHok.defmodule RayMarching do
  # Magnitude de um vetor 3D
  defd lenght(v_x, v_y, v_z) do
    return(sqrt(v_x * v_x + v_y * v_y + v_z * v_z))
  end

  # Sphere Signed Distance Function
  defd sphere_sdf(p_x, p_y, p_z, center_x, center_y, center_z, radius) do
    # Calcula o vetor do ponto p para o centro da esfera
    dx = p_x - center_x
    dy = p_y - center_y
    dz = p_z - center_z

    # A distância do ponto p para a superfície da esfera é a magnitude do vetor menos o raio
    return(lenght(dx, dy, dz) - radius)
  end

  # 3D map of the scene, returns the distance to the closest object
  defd scene_sdf(p_x, p_y, p_z) do
    # For testing, I will just a single sphere in the center of the scene
    s1_x = 0.0
    s1_y = 0.0
    s1_z = 0.0
    s1_radius = 1.0

    return(sphere_sdf(p_x, p_y, p_z, s1_x, s1_y, s1_z, s1_radius))
  end

  defk render_kernel(img_array, dim_x, dim_y) do
    tid_x = get_global_id(0)
    tid_y = get_global_id(1)

    p_x = 0.0
    p_y = 1.0
    p_z = -1.0

    if tid_x < dim_x && tid_y < dim_y do
      # foo = scene_sdf(p_x, p_y, p_z)
      foo = -1.0

      if foo < 0.0 do
        # Se o ponto está dentro da esfera, pinta de vermelho
        img_array[tid_y * dim_x + tid_x] = 0x0000FF
      else
        # Se o ponto está fora da esfera, pinta de preto
        img_array[tid_y * dim_x + tid_x] = 0x000000
      end
    end
  end

  defp render_worker(sdl_pid, gpu_array, dim_x, dim_y) do
    threads_per_block = 16
    blocks_x = div(dim_x + threads_per_block - 1, threads_per_block)
    blocks_y = div(dim_y + threads_per_block - 1, threads_per_block)

    # Lançando o kernel de renderização na GPU
    OCLPolyHok.spawn(
      &RayMarching.render_kernel/3,
      {blocks_x, blocks_y, 1},
      {threads_per_block, threads_per_block, 1},
      [gpu_array, dim_x, dim_y]
    )

    # Pega o resultado da renderização na GPU e envia para o processo do SDL2 para atualizar a textura
    result_ram = gpu_array |> OCLPolyHok.get_gnx()
    send(sdl_pid, {:render_img, result_ram})

    # Recursão para continuar o loop de renderização
    render_worker(sdl_pid, gpu_array, dim_x, dim_y)
  end

  def render(sdl_pid, gpu_array, dim_x, dim_y) do
    render_worker_pid = spawn_link(fn -> render_worker(sdl_pid, gpu_array, dim_x, dim_y) end)

    sdl_monitor_ref = Process.monitor(sdl_pid)

    receive do
      {:DOWN, ^sdl_monitor_ref, :process, _pid, _reason} ->
        # Se o processo do SDL for encerrado, saímos do loop de renderização lindamente =)
        IO.puts("[RayMarching] SDL process has terminated. Stopping render loop.")

        Process.demonitor(sdl_monitor_ref, [:flush])

        # Unlink and kill render worker to ensure it doesn't keep running in the background
        Process.unlink(render_worker_pid)
        Process.exit(render_worker_pid, :kill)

        :ok
    end
  end
end

# Estrutura do programa:
# * um processo vai cuidar da janela -> SDL2
# * outro processo (o principal) da renderização -> RayMarching

window_width = 500
window_height = 500

# Array com a imagem na GPU - global
img_array_gpu = OCLPolyHok.new_gnx(window_width, window_height, {:s, 32})

# Lançando o processo do SDL2
SDL2.create_window_nif(~c"teste", window_width, window_height)
sdl_process_pid = spawn(fn -> SDL2.loop() end)

# Processo principal cuida da renderização, e envia a imagem renderizada para o processo do SDL2
RayMarching.render(sdl_process_pid, img_array_gpu, window_width, window_height)
