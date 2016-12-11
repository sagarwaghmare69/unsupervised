--[[
   Latent Semantic Analysis. Primaryly used in Document Analysis.
   LSA is similar to PCA-SVD. It used SVD for computing the eigen
   values and vectors. Another difference is that the projections
   are scaled such that it has unit covariance.

   Ref: 15.4 Bayesian Reasoning and Machine Learning
--]]

local LSA, parent = torch.class("unsupervised.LSA", "unsupervised.PCA")

-- If rescale is false then this is same as PCA-SVD
function LSA:__init(M, rescale)
   parent.__init(self, M)
   self.rescale = rescale or true -- Default rescale projections
end

function LSA:__init(X, inplace)

end
