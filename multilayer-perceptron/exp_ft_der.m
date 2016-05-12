
function ret = exp_ft_der(h, b)
    ret = 2 * b * h.*(1 - h);
end
