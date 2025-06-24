function M = powerS( X,maxIter )

    [u,s,v] = svd(X);
    s(maxIter:end,maxIter:end) = 0;
    M = u*s*v';
    
end

