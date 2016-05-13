
% example of use: test(@xor, @exp_ft, @exp_ft_der, 0.5, 2, false)
function ret = test(ft, g, g_der, n, b, incremental)
    inputs = [1 1 1; 1 0 1; 0 1 1; 0 0 1; 0 1 0; 1 0 0; 1 1 0; 0 0 0];
    % inputs = [1 1; 1 0; 0 1; 0 0];
    s = ft(inputs);

    t{1} = inputs;
    t{2} = s;

    err = 0.001;

    nets = generate_nets([3 5 2 1]);

    if (incremental)
        resolved_nets = multilayer_perceptron_incremental(nets, t, err, g, g_der, n, b);
    else
        resolved_nets = multilayer_perceptron_batch(nets, t, err, g, g_der, n, b);
    endif

    layer_outputs = forward_step(inputs, resolved_nets, g, b);

    ret = layer_outputs(size(resolved_nets)(2));
end
