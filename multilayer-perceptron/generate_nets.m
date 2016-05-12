
function nets = generate_nets(neurons_per_layer)

    for i = 1: (columns(neurons_per_layer) - 1)

        in_layer_count = neurons_per_layer(i) + 1;
        out_layer_count = neurons_per_layer(i + 1);

        % numbers will be between -0.5 and 0.5
        nets{i} = rand(in_layer_count, out_layer_count).-0.5;

    end

end
