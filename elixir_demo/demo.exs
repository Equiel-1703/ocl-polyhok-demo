require OCLPolyHok

# OCLPolyHok.set_debug_logs(true)
# OCLPolyHok.TypeInference.set_debug_logs(true)

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
        # Se não houve pedido de encerramento, simplesmente continuamos o loop
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

  def update_image_nif(_img_array) do
    :erlang.nif_error(:nif_not_implemented)
  end
end

OCLPolyHok.defmodule RayMarching do
  # Magnitude de um vetor 3D
  defd lenght(v_x, v_y, v_z) do
    return(sqrt(v_x * v_x + v_y * v_y + v_z * v_z))
  end

  # Cálculo de distancia euclidiana entre dois pontos 3D
  defd distance(p1_x, p1_y, p1_z, p2_x, p2_y, p2_z) do
    dx = p1_x - p2_x
    dy = p1_y - p2_y
    dz = p1_z - p2_z
    return(sqrt(dx * dx + dy * dy + dz * dz))
  end

  # Produto escalar entre dois vetores 3D
  defd dot(v1_x, v1_y, v1_z, v2_x, v2_y, v2_z) do
    return(1.0 * (v1_x * v2_x + v1_y * v2_y + v1_z * v2_z))
  end

  # Sphere Signed Distance Function (SDF)
  # A distância de um ponto p até a superfície de uma esfera é a distância do ponto p
  # para o centro da esfera menos o raio da esfera.
  # Os argumentos dessa função são um ponto p (p_x, p_y, p_z) e os parâmetros da esfera (center_x, center_y, center_z, radius).
  defd sphere_sdf(p_x, p_y, p_z, center_x, center_y, center_z, radius) do
    # Calcula a distância do ponto p para o centro da esfera usando a função de distância euclidiana
    l = distance(p_x, p_y, p_z, center_x, center_y, center_z)
    # Subtrai o raio da esfera
    return(l - radius)
  end

  # 3D map of the scene, returns the distance to the closest object
  # Receives a point p (p_x, p_y, p_z) and returns the distance to the closest object in the scene
  # to that point
  defd scene_sdf(p_x, p_y, p_z, time) do
    # Sphere 1
    s1_x = 0.0 * time
    s1_z = -1.0
    s1_radius = 0.5

    # Sphere 1 animation: it bounces!
    height = 0.5
    s1_y = -fabs(sin(time * 5.0) * height)

    return(sphere_sdf(p_x, p_y, p_z, s1_x, s1_y, s1_z, s1_radius))
  end

  # Ray marching function
  # Receives a ray origin (ray_origin_x, ray_origin_y, ray_origin_z) and a ray direction (ray_dir_x, ray_dir_y, ray_dir_z)
  # and returns the distance traveled along the ray until it hits an object or reaches a maximum distance
  defd ray_march(ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x, ray_dir_y, ray_dir_z, time) do
    total_distance = 0.0

    # Far clip plane
    max_distance = 100.0

    max_steps = 64

    # Minimum distance to consider a hit
    min_distance = 0.001

    for i in range(0, max_steps) do
      # Calculate the current point along the ray
      current_x = ray_origin_x + ray_dir_x * total_distance
      current_y = ray_origin_y + ray_dir_y * total_distance
      current_z = ray_origin_z + ray_dir_z * total_distance

      # Get the distance to the closest object from the current point
      dist_to_closest = scene_sdf(current_x, current_y, current_z, time)

      if dist_to_closest < min_distance do
        # Hit an object
        return(total_distance)
      end

      if total_distance > max_distance do
        # Exceeded maximum distance without hitting anything
        # Indicate no hit with a negative value
        return(-1.0)
      end

      # Move along the ray by the distance to the closest object
      total_distance = total_distance + dist_to_closest
    end

    # If we perform the maximum number of steps without hitting anything, we also consider it a miss
    return(-1.0)
  end

  defk render_kernel(img_array, dim_x, dim_y, time) do
    tid_x = get_global_id(0)
    tid_y = get_global_id(1)

    if tid_x < dim_x && tid_y < dim_y do
      # Creating UV coordinates
      u = 2.0 * (1.0 * tid_x / (1.0 * dim_x)) - 1.0
      v = 2.0 * (1.0 * tid_y / (1.0 * dim_y)) - 1.0
      # Aspect ratio correction
      u = u * (1.0 * dim_x / (1.0 * dim_y))

      # Set up camera origin (location)
      cam_x = 0.0
      cam_y = 0.0
      cam_z = 0.0

      # Ray direction (where the camera is looking at)
      # The camera is looking towards the negative z direction
      ray_dir_x = u - cam_x
      ray_dir_y = v - cam_y
      ray_dir_z = -1.0 - cam_z

      # Normalize ray direction
      ray_dir_len = lenght(ray_dir_x, ray_dir_y, ray_dir_z)
      ray_dir_x = ray_dir_x / ray_dir_len
      ray_dir_y = ray_dir_y / ray_dir_len
      ray_dir_z = ray_dir_z / ray_dir_len

      # Perform ray marching
      dist = ray_march(cam_x, cam_y, cam_z, ray_dir_x, ray_dir_y, ray_dir_z, time)

      if dist > 0.0 do
        # We hit something!
        # Calculate the point of intersection
        hit_x = cam_x + ray_dir_x * dist
        hit_y = cam_y + ray_dir_y * dist
        hit_z = cam_z + ray_dir_z * dist

        # Let's do some lighting calculation =D
        # I'll add a simple directional light pointing to (0, 1, -1)
        light_dir_x = 0.0
        light_dir_y = 1.0
        light_dir_z = -1.0
        # Normalize light direction
        light_dir_len = lenght(light_dir_x, light_dir_y, light_dir_z)
        light_dir_x = light_dir_x / light_dir_len
        light_dir_y = light_dir_y / light_dir_len
        light_dir_z = light_dir_z / light_dir_len

        # Calculate the normal at the hit point using the gradient of the SDF
        epsilon = 0.001
        n_x = scene_sdf(hit_x + epsilon, hit_y, hit_z, time) - scene_sdf(hit_x - epsilon, hit_y, hit_z, time)
        n_y = scene_sdf(hit_x, hit_y + epsilon, hit_z, time) - scene_sdf(hit_x, hit_y - epsilon, hit_z, time)
        n_z = scene_sdf(hit_x, hit_y, hit_z + epsilon, time) - scene_sdf(hit_x, hit_y, hit_z - epsilon, time)
        # Normalize the normal
        n_len = lenght(n_x, n_y, n_z)
        n_x = n_x / n_len
        n_y = n_y / n_len
        n_z = n_z / n_len

        # Calculate the diffuse lighting using the dot product between the normal and the light direction
        brightness = -dot(n_x, n_y, n_z, light_dir_x, light_dir_y, light_dir_z)
        # Clamp to [0, 1]
        brightness = min(max(brightness, 0.0), 1.0)

        # Color phasing, get remainder of time divided by 255 to create a cycling effect
        color = fmod(time * 100.0, 65536.0)

        # Apply the brightness to a base color (I'm using blue)
        img_array[tid_y * dim_x + tid_x] = trunc(brightness * color)
      else
        # If we didn't hit anything, set the pixel to black
        img_array[tid_y * dim_x + tid_x] = 0x000000
      end
    end
  end

  defk test_time_kernel(time_array, len_x, len_y, time) do
    tid_x = get_global_id(0)
    tid_y = get_global_id(1)

    if tid_x < len_x && tid_y < len_y do
      idx = tid_y * len_x + tid_x
      color = trunc(fmod(time * 100.0, 65536.0))
      time_array[idx] = color
    end
  end

  defp render_worker(sdl_pid, gpu_array, dim_x, dim_y) do
    threads_per_block = 16
    blocks_x = div(dim_x + threads_per_block - 1, threads_per_block)
    blocks_y = div(dim_y + threads_per_block - 1, threads_per_block)

    {time, _} = :erlang.statistics(:wall_clock)
    time = time / 1000.0

    # Lançando o kernel de renderização na GPU
    OCLPolyHok.spawn(
      &RayMarching.render_kernel/4,
      {blocks_x, blocks_y, 1},
      {threads_per_block, threads_per_block, 1},
      [gpu_array, dim_x, dim_y, time]
    )

    # Timing debug kernel
    # OCLPolyHok.spawn(
    #   &RayMarching.test_time_kernel/4,
    #   {blocks_x, blocks_y, 1},
    #   {threads_per_block, threads_per_block, 1},
    #   [gpu_array, dim_x, dim_y, time]
    # )

    # Pega o resultado da renderização na GPU e envia para o processo do SDL2 para atualizar a textura
    result_ram = gpu_array |> OCLPolyHok.get_gnx()

    # IO.inspect(result_ram, label: "result_ram")

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
SDL2.create_window_nif(~c"OCL-PolyHok - Ray Marching Demo", window_width, window_height)
sdl_process_pid = spawn(fn -> SDL2.loop() end)

# Processo principal cuida da renderização, e envia a imagem renderizada para o processo do SDL2
RayMarching.render(sdl_process_pid, img_array_gpu, window_width, window_height)

# Add small timer sleep here to ensure cleanup messages are printed
:timer.sleep(100)
