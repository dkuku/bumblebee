defmodule Bumblebee.Audio.ChunkingTest do
  use ExUnit.Case, async: true

  test "chunks a tensor stream with overlap and preserves a final partial chunk" do
    chunks =
      Bumblebee.Audio.chunk_audio(
        [Nx.tensor([0, 1]), Nx.tensor([2, 3, 4]), Nx.tensor([5, 6, 7, 8])],
        1,
        4,
        1
      )
      |> Enum.map(&Nx.to_list/1)

    assert chunks == [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8]]
  end

  test "rejects context that leaves no forward progress" do
    assert_raise ArgumentError,
                 ":chunk_num_seconds must be more than double the length of :context_num_seconds",
                 fn ->
                   Bumblebee.Audio.chunk_audio([Nx.tensor([0, 1])], 1, 2, 1)
                   |> Enum.to_list()
                 end
  end
end
