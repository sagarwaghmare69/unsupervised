--[[
   Factor Analysis.
   FA model is a classical statistical model and is essentially
   probabilistic PCA. It can also form simple low dimensional
   generative model.
   Ref: Algorithm 21.1 Bayesian Reasoning and Machine Learning
--]]

local FA, parent = torch.class('unsupervised.FA', 'unsupervised.ParentModule')

function FA:__init(H)
   assert(H ~= nil, "H must be a positive integer.")
   parent.__init(self)
   self.H = H
end

-- Input NxD dimensional tensor
-- Max number of iterations
-- If diff between consecutive LL is < minDiffLL then stop.
function FA:train(X, maxIter, useIter, minDiffLL, inplace)
   self.N = X:size(1)
   self.D = X:size(2)
   local maxIter = maxIter or 100
   local useIter = useIter or false
   local minDiffLL = minDiffLL or 1e-05
   local inplace = inplace or false

   -- Initialize noise diagonal covariance
   local psi = torch.ones(self.D)
   
   self.mean = X:mean(1)
   self.variance = X:var(1)

   -- Centered X
   local normedX = self:_meanZero(X, self.mean, inplace)

   local iter = 0
   local psiHalf = psi.new():resizeAs(psi)
   local psiInvHalf = psi.new():resizeAs(psi)
   local diagPsiInvHalf = torch.diag(psi)
   local scaledX = normedX.new():resizeAs(normedX)
   local U = normedX.new()
   local Uh = normedX.new()
   local S = normedX.new()
   local W = normedX.new()
   local lambda = normedX.new()
   local lambdaH = normedX.new()
   local lambdaF = normedX.new()
   while(true) do
      psiInvHalf:copy(psi):pow(-0.5)
      diagPsiInvHalf:diag(psiInvHalf)
      scaledX:mm(normedX, diagPsiInvHalf):div(math.sqrt(self.N))
      torch.svd(U, S, W, scaledX)
      lambda:resizeAs(S):copy(S):pow(2) -- Eigen values
      Uh = U[{{1, self.H}}]
      lambdaH = lambda[{{1, H}}]
      psiHalf:copy(psi):pow(0.5)
      lambdaF:resizeAs(lambdaH):copy(lambdaH):add(-1):pow(0.5)
   end
end
