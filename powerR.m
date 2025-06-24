function M = powerR( X )

    [n, m] = size(X);
    
    z = ones(m, 1);
    
    y = X * z;
    y = y/norm(y,2);
    
    tmp = X' * y;
    y = X * tmp;
    normy = norm(y,2);
    if normy ~= 0
        y = y/normy;      
    end
    
    v = X' * y;
    s = norm(v,2);
    u = y;
    M = u*v';

end

