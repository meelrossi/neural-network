
% example of use: test(@xor, @exp_ft, @exp_ft_der, 0.5, 2, false)
% example of use: test(@xor, @tanh_ft, @tanh_ft_der, 0.5, 1, true)
% example of use: test(@xor, @tanh_ft, @tanh_ft_der, 0.2, 0.5, false)
function ret = test(ft, g, g_der, n, betha, incremental, momentum)
    inputs = [1 1 1; 1 0 1; 0 1 1; 0 0 1; 0 1 0; 1 0 0; 1 1 0; 0 0 0];
    % inputs = [1 1; 1 0; 0 1; 0 0];
    s = ft(inputs);

    t{1} = inputs;
    t{2} = s;

    err = 0.001;

    nets = generate_nets([3 5 2 1]);

    if (incremental)
        if (momentum)
            resolved_nets = multilayer_perceptron_incremental_momentum(nets, t, err, g, g_der, n, betha, 0.9);
        else
            resolved_nets = multilayer_perceptron_incremental(nets, t, err, g, g_der, n, betha);
        endif
    else
        if (momentum)
            resolved_nets = multilayer_perceptron_batch_momentum(nets, t, err, g, g_der, n, betha, 0.9);
        else
            resolved_nets = multilayer_perceptron_batch(nets, t, err, g, g_der, n, betha);
        endif
    endif

    layer_outputs = forward_step(inputs, resolved_nets, g, betha);

    ret = layer_outputs(size(resolved_nets)(2));
end
