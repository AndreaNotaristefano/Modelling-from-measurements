function [Phi, Lambda, b, Vr] = DMD(X,X1,r)

[U, Sigma, V] = svd(X, 'econ');
if r=='max'
    r = length(U(1,:));
end
Ur = U(:,1:r);
Sigmar = Sigma(1:r,1:r);
Vr = V(:,1:r);

Atilde = Ur' * X1 * Vr / Sigmar;
[W, Lambda] = eig(Atilde); % Lambda eigenvalues, W eigenvectors

Phi = X1*(Vr/Sigmar)*W;
alpha1 = Sigmar*Vr(1,:)';
b = (W*Lambda)\alpha1;

end

