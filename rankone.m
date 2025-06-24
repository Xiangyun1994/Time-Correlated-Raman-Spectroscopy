function B = rankone( A,maxIter,threshold )

    B = zeros(size(A));

   
   
    for i = 1:maxIter

        d = A-B;
        S = powerR(d);
        B = B + S;

        x = sum(abs(S(:)))./sum(abs(A(:)))

        if x < threshold
            disp('Reached threshold! Exiting loop...');
        break;
  
    end
    
end

