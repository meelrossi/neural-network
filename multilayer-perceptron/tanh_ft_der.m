
function ret = tanh_ft_der(h, b)
    ret = b * (1 - h.^2);
end
