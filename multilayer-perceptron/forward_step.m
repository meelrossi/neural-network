

function ret = forward_step(inputs, nets, g, b)
    inputs_count = rows(inputs);
    nets_count = size(nets)(2);

    % forward step
    in_layer = inputs;
    for i = 1 : nets_count
        layer = [ones(inputs_count, 1).*(-1) in_layer];
        V{i} = g(layer * nets{i}, b);
        in_layer = V{i};
    end

    ret = V;
 end
 