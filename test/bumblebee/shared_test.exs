defmodule Bumblebee.SharedTest do
  use ExUnit.Case, async: true

  alias Bumblebee.Shared

  describe "validate_label_options/1" do
    test "passes when :id_to_label is empty" do
      spec = %{__struct__: TestConfig, num_labels: 3, id_to_label: %{}}

      assert Shared.validate_label_options(spec) == spec
    end

    test "passes when :id_to_label is matches :num_labels" do
      id_to_label = %{0 => "cat", 1 => "dog", 2 => "squirrel"}
      spec = %{__struct__: TestConfig, num_labels: 3, id_to_label: id_to_label}

      assert Shared.validate_label_options(spec) == spec
    end

    test "raises an error if mismatched :num_labels and :id_to_label are given" do
      id_to_label = %{0 => "cat", 1 => "dog"}
      spec = %{__struct__: TestConfig, num_labels: 3, id_to_label: id_to_label}

      assert_raise ArgumentError,
                   ~s/size mismatch between :num_labels (3) and :id_to_label (%{0 => "cat", 1 => "dog"})/,
                   fn ->
                     Shared.validate_label_options(spec)
                   end
    end
  end

  describe "common_options_from_transformers/2" do
    test "loads use_bidirectional_attention from is_causal: false" do
      spec = %{use_bidirectional_attention: false}
      data = %{"is_causal" => false}

      assert Shared.common_options_from_transformers(data, spec) == [
               use_bidirectional_attention: true
             ]
    end

    test "loads use_bidirectional_attention when explicitly true" do
      spec = %{use_bidirectional_attention: false}
      data = %{"use_bidirectional_attention" => true}

      assert Shared.common_options_from_transformers(data, spec) == [
               use_bidirectional_attention: true
             ]
    end

    test "raises when is_causal and use_bidirectional_attention conflict" do
      spec = %{use_bidirectional_attention: false}

      assert_raise ArgumentError,
                   ~s/conflicting configuration: "is_causal" is true, but "use_bidirectional_attention" is true/,
                   fn ->
                     Shared.common_options_from_transformers(
                       %{"is_causal" => true, "use_bidirectional_attention" => true},
                       spec
                     )
                   end

      assert_raise ArgumentError,
                   ~s/conflicting configuration: "is_causal" is false, but "use_bidirectional_attention" is false/,
                   fn ->
                     Shared.common_options_from_transformers(
                       %{"is_causal" => false, "use_bidirectional_attention" => false},
                       spec
                     )
                   end
    end
  end
end
