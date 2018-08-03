function setup_LGBWT()
%SETUP_LGBWT Sets up the code for Learning Grimaces By Watching TV, by adding
% its folders to the Matlab path
%
% Copyright (C) 2016 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/core'], [root '/facevalue-exps'], [root '/figs']) ;
  addpath([root '/sfew-exps'], [root '/fer-exps']) ;
