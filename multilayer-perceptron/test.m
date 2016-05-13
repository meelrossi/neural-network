
function ret = test(ft, g, g_der, n, b)
    inputs = [1 1; 1 0; 0 1; 0 0];
    s = ft(inputs);

    t{1} = inputs;
    t{2} = s;

    n = 0.5;
    b = 2;
    err = 0.001;

    nets = generate_nets([2 5 3 1]);

    resolved_nets = multilayer_perceptron_batch(nets, t, err, g, g_der, n, b);


    layer_outputs = forward_step(inputs, resolved_nets, g, b);

    ret = layer_outputs(size(resolved_nets)(2));
end
