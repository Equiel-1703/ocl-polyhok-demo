defmodule SDL2 do
  @on_load :load_nifs
  def load_nifs do
    :erlang.load_nif(~c"priv/demo_nifs", 0)
  end

  def loop() do
    case close_requested_nif() do
      true ->
        :ok

      false ->
        render_window_nif()
        loop()
    end
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
end

SDL2.create_window_nif(~c"teste", 500, 500)
SDL2.loop()
