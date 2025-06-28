import gtsam
import gtsam.noiseModel
import numpy as np

def bundle_adjustment(poses, points, features, observations, reprojection_threshold=5e-3):
    #bundle adjustment using gtsam
    #create factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    fix_first = False
    #add camera poses
    for i, pose in enumerate(poses):
        if pose is None:
            continue
        pose = gtsam.Pose3(gtsam.Rot3(pose[:3, :3]), gtsam.Point3(pose[:3, 3]))
        initial_estimate.insert(gtsam.symbol('p', i), pose.inverse())
        if fix_first:
            graph.add(gtsam.PriorFactorPose3(gtsam.symbol('p', i), gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]))))
            fix_first = False
        

    #add 3d points
    for i, point in enumerate(points):
        point = gtsam.Point3(point)
        initial_estimate.insert(gtsam.symbol('l', i), point)

    #add reprojection factors
    #TODO:  добавьте все наблюдения в GT-SAM граф
    # вам нужно добавить gtsam.GenericProjectionFactorCal3_S2 для каждого наблюдения
    # каждый фактор принимает следующие аргументы:
    # 1. Наблюдаемая точка в виде gtsam.Point2
    # 2. Модель шума
    # 3. Символ камеры, например gtsam.symbol('p', camera_id)
    # 4. Символ точки, например gtsam.symbol('l', point_id)
    # 5. Модель камеры, как gtsam.Cal3_S2(fx, fy, s, u0, v0)
    # где fx и fy - фокусные расстояния, s - skew, u0 и v0 - координаты центра проекции

    noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    huber = gtsam.noiseModel.mEstimator.Huber(reprojection_threshold)
    noise = gtsam.noiseModel.Robust(huber, noise)

    #--- Пример ответа --- #

    for point_id, observarions in enumerate(observations):
        for obs in observarions:
            camera_id = obs[0]
            feature_id = obs[1]
            point2d = features[camera_id][0][feature_id]

            graph.add(gtsam.GenericProjectionFactorCal3_S2(gtsam.Point2(point2d[0], point2d[1]), noise, gtsam.symbol('p', camera_id), gtsam.symbol('l', point_id), gtsam.Cal3_S2(1, 1, 0, 0, 0)))

    #--- Конеч примера --- #
        
    #optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    params.setlambdaInitial(1e-3)
    params.setMaxIterations(10)
    params.setlambdaUpperBound(1e7)
    params.setlambdaLowerBound(1e-7)
    params.setRelativeErrorTol(1e-5)

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()

    #update poses and points
    for i in range(len(poses)):
        if poses[i] is not None:
            poses[i] = np.eye(4) 
            pose = result.atPose3(gtsam.symbol('p', i)).inverse()
            poses[i][:3, :3] = pose.rotation().matrix()
            poses[i][:3, 3] = pose.translation()
        
    for i in range(len(points)):
        point = result.atPoint3((gtsam.symbol('l', i)))
        points[i] = point