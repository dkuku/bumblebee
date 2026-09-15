defmodule Bumblebee.Audio.Qwen3ASRFeaturizerTest do
  use ExUnit.Case, async: true

  test "pads waveforms and returns Qwen layout" do
    featurizer = Bumblebee.configure(Bumblebee.Audio.Qwen3ASRFeaturizer)
    audio = Nx.iota({16_000}, type: :f32)

    assert %{"input_features" => features} = Bumblebee.apply_featurizer(featurizer, audio)
    assert Nx.shape(features) == {1, 128, 3000}
  end

  test "rejects waveforms longer than the configured duration" do
    featurizer = Bumblebee.configure(Bumblebee.Audio.Qwen3ASRFeaturizer, num_seconds: 1)

    assert_raise ArgumentError, ~r/at most 1 seconds/, fn ->
      Bumblebee.Audio.Qwen3ASRFeaturizer.process_input(featurizer, Nx.iota({16_001}))
    end
  end
end
