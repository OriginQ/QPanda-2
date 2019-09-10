import os

currentdir = os.path.abspath(__file__)
currentdir = os.path.dirname(currentdir) + r"\ChemiQCalc.exe"

for iter in [10000]:
  for distance in [1.0,1.1,1.2,1.3,1.4, 1.5, 1.6,1.7,1.8,1.9,2.0]:

    optimizer_type = 'NELDER_MEAD'
    transform_type = 'JW'
    basis = 'sto-3g'
    ucctype = 'UCCSD'
    backend = 'CPU_SINGLE_THREAD'
    psi4dir = 'D:/ChemiQ_release'
    
    datadir = 'LiH-{_distance}-iter-{_iter}-{_optimizer_type}-{_transform_type}-{_ucctype}'.format(
        _iter=iter, 
        _distance=distance, 
        _transform_type=transform_type, 
        _optimizer_type = optimizer_type,
        _ucctype = ucctype
    )
    print(
        'Start-Job -ScriptBlock {{mkdir {_datadir}; & "{_currentdir}" --molecule "H 0 0 0`nLi 0 0 {_distance}" --iters {_iter} --datadir "{_datadir}" --optimizer {_optimizer_type} --transform {_transform_type} --basis {_basis} --UCC {_ucctype} --backend {_backend} --psi4dir "{_psi4dir}" }}'.format(
        _iter=iter,
        _distance=distance, 
        _datadir = datadir,
        _currentdir = currentdir,
        _optimizer_type = optimizer_type,
        _transform_type = transform_type,
        _basis = basis,
        _ucctype = ucctype,
        _backend = backend,
        _psi4dir = psi4dir
        )
    )
