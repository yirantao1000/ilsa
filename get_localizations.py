import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw
from cv2 import aruco as ar
import pyrealsense2 as rs
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

init_pos = 

def get_objects(task):
    if task == 'cereal':
        obj_prompts = ['cup', 'bowl']
        orientation_prompts = ['small table']
        processes = ['pick', 'pour'] #has to be in order
        above_bowl = 
    elif task == 'pill':
        obj_prompts = ['small round red button', 'small pill cup']
        orientation_prompts = ['drawer', 'small round red button']
        processes = ['open', 'pick', 'throw'] #has to be in order
        gripper_back = 
        out_ori_button = 
        above_button = 


    image_save_dir = f'./data/{task}/incremental/{interact_epoch}/images/{interact_index}'
    assert not os.path.exists(image_save_dir)
    os.makedirs(image_save_dir)
    #ar tag locations for localization
    marker_y = 0.185
    marker_x = 0.275

    sam_dir = ''
    if sam_dir not in sys.path:
        sys.path.append(sam_dir)
    from lang_sam import LangSAM
    sam_cpu = False
    lang_model = LangSAM(cpu=sam_cpu)

    def find_closest_element(target, elements):
        return min(elements, key=lambda x: abs(x - target))
    def find_top_point(mask):
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        indices = np.argwhere(mask)
        assert indices > 0
        return np.min(indices[:, 0])

    def visualize_masks(image_pil, masks):
        def calculate_centroid(mask):
            indices = np.argwhere(mask)
            if len(indices) == 0:
                return None
            centroid = np.mean(indices, axis=0)
            return int(centroid[1]), int(centroid[0])  # return (x, y)
        
        image_np = np.array(image_pil)

        fig, ax = plt.subplots()
        ax.imshow(image_np)

        for mask in masks:
            # ax.imshow(np.ma.masked_where(~mask, mask), alpha=0.5, cmap='jet')
            centroid = calculate_centroid(mask.numpy())
            if centroid:
                # centroids.append(centroid)
                ax.plot(centroid[0], centroid[1], 'ro')

        clicked_mask = [None]

        def on_click(event):
            if event.inaxes is not None:
                x, y = int(event.xdata), int(event.ydata)
                if x >= 0 and y >= 0 and x < image_np.shape[1] and y < image_np.shape[0]:
                    for i, mask in enumerate(masks):
                        if mask[y, x]:
                            clicked_mask[0] = mask
                            print(f'Mask {i} clicked')
                            fig.canvas.mpl_disconnect(cid)
                            plt.close(fig)
                            break

        cid = fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show(block=True)

        return clicked_mask[0]

    def find_closest_point_not_in_indices(centroid, x_indices, y_indices, mask_shape):
        centroid_x, centroid_y = centroid
        rows, cols = mask_shape
        
        original_points = set(zip(x_indices, y_indices))
        
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        
        radius = 1

        while radius < max(rows, cols):
            for dx, dy in directions:
                for step in range(-radius, radius + 1):
                    nx, ny = centroid_x + step * dx, centroid_y + step * dy
                    if 0 <= nx < cols and 0 <= ny < rows:  # Check bounds
                        if (nx, ny) not in original_points:
                            distance = np.sqrt((nx - centroid_x)**2 + (ny - centroid_y)**2)
                            return nx, ny, distance
            radius += 1
        
        return None

    def get_mask(text_prompt, image_path, return_mask = False):
        print(text_prompt)
        # device = torch.device("cpu")
        # model.to(device)
        image_pil = Image.open(image_path).convert("RGB")
        # print(image_pil.size)           

        masks, _, _, _ = lang_model.predict(image_pil, text_prompt)
        if 'drawer' in text_prompt:
            mask = masks[0]
        else:
            print(f"Number of detected {text_prompt}: {masks.shape[0]}")
            mask = visualize_masks(image_pil, masks)
        # print(mask)
        # exit()

        # mask_np = masks[0].numpy()
        mask_np = mask.numpy()
        
        image_np = np.array(image_pil)

        y_indices, x_indices = np.where(mask_np)

        # Calculate the centroid (middle point)
        if len(y_indices) > 0 and len(x_indices) > 0:
            centroid_y = int(np.median(y_indices))
            centroid_x = int(np.median(x_indices))
            centroid = (centroid_x, centroid_y)

            if task == 'cereal' and text_prompt in obj_prompts:
                if obj_prompts.index(text_prompt) == 0:
                    _, _, distance_from_non_cup = find_closest_point_not_in_indices(centroid, x_indices, y_indices, mask_np.shape) 
                    print("distance from non cup:", distance_from_non_cup)
            assert (centroid_y in y_indices) and (centroid_x in x_indices)
            
            if 'bowl' in text_prompt:
                centroid_y+=10
                centroid = (centroid_x, centroid_y)


            horizontal_mask_indices = np.where(y_indices == centroid_y)[0]
            assert len(horizontal_mask_indices) > 0               
            leftmost_x = np.min(x_indices[horizontal_mask_indices])
            rightmost_x = np.max(x_indices[horizontal_mask_indices])

            leftmost = (leftmost_x, centroid_y)
            rightmost = (rightmost_x, centroid_y)
            
        else:
            centroid = None

        image_rgba = np.concatenate([image_np, np.full((image_np.shape[0], image_np.shape[1], 1), 255, dtype=np.uint8)], axis=2)
        overlay = Image.new("RGBA", image_pil.size, (255, 0, 0, 0))
        mask_color = np.array([255, 0, 0, 128], dtype=np.uint8)  # Red with 50% opacity

        # Apply the mask: wherever mask_np is True, replace the pixel with a blend of the image and mask color
        masked_pixels = (mask_color[:3] * 0.5 + image_rgba[:, :, :3] * 0.5).astype(np.uint8)
        image_rgba[mask_np, :3] = masked_pixels[mask_np]
        image_rgba[mask_np, 3] = mask_color[3]  # Adjust alpha where mask is True


        final_image = Image.fromarray(image_rgba, 'RGBA')


        if centroid is not None:
            draw = ImageDraw.Draw(final_image)
            radius = 5
            color = (255, 0, 0)  
            draw.ellipse(
                [
                    (centroid[0] - radius, centroid[1] - radius),
                    (centroid[0] + radius, centroid[1] + radius),
                ],
                outline=color,
                width=3,
            )
            draw.ellipse(
                [
                    (leftmost[0] - radius, leftmost[1] - radius),
                    (leftmost[0] + radius, leftmost[1] + radius),
                ],
                outline=(0, 255, 0),  # Green
                width=3,
            )
            draw.ellipse(
                [
                    (rightmost[0] - radius, rightmost[1] - radius),
                    (rightmost[0] + radius, rightmost[1] + radius),
                ],
                outline=(0, 255, 0),  # Green
                width=3,
            )

        final_image.save(os.path.join(image_save_dir, f"./masked_image_{text_prompt}.png"))
        plt.close()
        if return_mask:
            return mask_np
        return centroid_x, centroid_y, leftmost_x, rightmost_x
    
        
    def compute_loc(prompts, image_path, known_dict = {}):    
        
        locs = {}
        for prompt in prompts:
            if prompt in known_dict.keys():
                x, y, leftmost_x, rightmost_x = known_dict[prompt]
            else:
                x, y, leftmost_x, rightmost_x = get_mask(prompt, image_path)
            assert rightmost_x>leftmost_x+4
            leftmost_x += 2
            rightmost_x -= 2
            depth_value = depth_image_copy[y, x] * depth_scale
            depth_value_leftmost = depth_image_copy[y, leftmost_x] * depth_scale
            depth_value_rightmost = depth_image_copy[y, rightmost_x] * depth_scale
            # print(f"Depth at ({x}, {y}): {depth_value}m")

            local_xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_value)
            local_xyz_leftmost = rs.rs2_deproject_pixel_to_point(intrinsics, [leftmost_x, y], depth_value_leftmost)
            local_xyz_rightmost = rs.rs2_deproject_pixel_to_point(intrinsics, [rightmost_x, y], depth_value_rightmost)
            # print("Local XYZ:", local_xyz)
            local_xyz_homogeneous = np.array([local_xyz[0], local_xyz[1], local_xyz[2], 1])
            local_xyz_homogeneous_leftmost = np.array([local_xyz_leftmost[0], local_xyz_leftmost[1], local_xyz_leftmost[2], 1])
            local_xyz_homogeneous_rightmost = np.array([local_xyz_rightmost[0], local_xyz_rightmost[1], local_xyz_rightmost[2], 1])

            global_xyz = extrinsic_matrix @ local_xyz_homogeneous
            global_xyz_leftmost = extrinsic_matrix @ local_xyz_homogeneous_leftmost
            global_xyz_rightmost = extrinsic_matrix @ local_xyz_homogeneous_rightmost
            assert global_xyz[3]==global_xyz_leftmost[3]==global_xyz_rightmost[3]==1
            # print(f"Aruco Marker Coordinate: {global_xyz}")

            global_xyz = world2marker @ global_xyz
            global_xyz_leftmost = world2marker @ global_xyz_leftmost
            global_xyz_rightmost = world2marker @ global_xyz_rightmost
          
            if global_xyz[1]<=0.1:
                print("adjusting left side")
                print("original: ", global_xyz)
                global_xyz[1] += -0.01
                global_xyz_leftmost[1] += -0.01
                global_xyz_rightmost[1] += -0.01
                print("adjusted: ", global_xyz)

            
            locs[prompt] = {}
            locs[prompt]['centroid_location'] = global_xyz[:-1]
            locs[prompt]['left_location'] = global_xyz_leftmost[:-1]
            locs[prompt]['right_location'] = global_xyz_rightmost[:-1]
        return locs

    def compute_orientation(prompts, image_path):
        
        def find_lowest_y(mask,x, image_pil, drawer_color):
            # image_pil = Image.open(image_path).convert("RGB")
            # for y in range(mask.shape[0]-1, -1, -1):
            #     if mask[y, x]:          
            #         r,g,b = image_pil.getpixel((x, y))
            #         if r > 100 and b > 100 and g < 100 and abs(r - b) < 50:
            #             return y
            for y in range(mask.shape[0]-1, -1, -1):
                if mask[y,x]:
                    color = image_pil.getpixel((x, y))
                    if np.all(np.isclose(color, drawer_color, atol=15)):
                        return y

        def find_highest_y(mask,x, image_path):
            for y in range(mask.shape[0]):
                if mask[y,x]:
                    return y

        def compute_global_xy(x,y):
            depth = depth_image_copy[y, x] * depth_scale
            local_xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
            
            local_xyz_homogeneous = np.array([local_xyz[0], local_xyz[1], local_xyz[2], 1])
            
            global_xyz = extrinsic_matrix @ local_xyz_homogeneous
            
            assert global_xyz[3]==1           
            global_xyz = world2marker @ global_xyz
            # print(global_xyz)
            return global_xyz[0], global_xyz[1]
            

        for prompt in prompts:
            if 'drawer' in prompt:
                drawer_mask = get_mask(prompt, image_path, return_mask = True)
                continue
            assert 'button' in prompt
            x, y, leftmost_x, rightmost_x = get_mask(prompt, image_path)
            button_x, button_y, button_leftmost_x, button_rightmost_x = x, y, leftmost_x, rightmost_x
            image_pil = Image.open(image_path).convert("RGB")
            drawer_color = image_pil.getpixel((x-2*(x-leftmost_x), y))
            print('drawer_color: ', drawer_color)
            xs = []
            ys = []
            sample_num = 5
            radius_ratio = 1
            xs.append(x)
            ys.append(find_lowest_y(drawer_mask, x, image_pil,drawer_color))
            for s in range(sample_num):
                s+=1
                left_x = x-s*int((x-leftmost_x)/radius_ratio)
                xs.append(left_x)
                lowest_y = find_lowest_y(drawer_mask, left_x, image_pil, drawer_color)
                # lowest_y -= 5
                # lowest_y = find_highest_y(drawer_mask, left_x, image_path)
                ys.append(lowest_y)
                right_x = x-s*int((x-rightmost_x)/radius_ratio)
                xs.append(right_x)
                lowest_y = find_lowest_y(drawer_mask, right_x, image_pil,drawer_color)
                # lowest_y -= 5
                # lowest_y = find_highest_y(drawer_mask, right_x, image_path)
                ys.append(lowest_y)
                # ys.append(find_lowest_y(drawer_mask, right_x, image_path))
            assert len(xs) == len(ys) == 2*sample_num+1
            xs = np.array(xs)
            ys = np.array(ys)

            # slope, intercept = np.polyfit(xs, ys, 1)

            image = Image.open(image_path)
            image_array = np.array(image)
            plt.figure()
            fig, ax = plt.subplots()
            ax.imshow(image_array)
            
            ax.scatter(xs, ys, color='blue', label='Data points',s=2)

            # best_fit_line = slope * xs + intercept
            # ax.plot(xs, best_fit_line, color='red', label='Best-fit line')

            plt.savefig(os.path.join(image_save_dir, f"./dawer_edge.png"))
            plt.close()


            xs_global = []
            ys_global = []

            for i in range(xs.shape[0]):
                try:
                    x_global, y_global = compute_global_xy(xs[i],ys[i])
                except:
                    continue
                xs_global.append(x_global)
                ys_global.append(y_global)
        
            xs_global = np.array(xs_global)
            ys_global = np.array(ys_global)

            xs_global_median = np.median(xs_global)
            abs_deviations = np.abs(xs_global - xs_global_median)
            # print("abs deviations: ", abs_deviations)
            threshold = 4 * np.median(abs_deviations)
            # print("filtering threshold: ", threshold)
            valid_mask = (abs_deviations <= threshold)
            # print("vaid mask: ", valid_mask)
            
            xs_global = xs_global[valid_mask]
            ys_global = ys_global[valid_mask]

            slope_global, intercept_global = np.polyfit(xs_global, ys_global, 1)
            best_fit_line_global = slope_global * xs_global + intercept_global

            plt.figure()

            plt.scatter(xs_global, ys_global, color='blue', label='Data points')
            plt.plot(xs_global, best_fit_line_global, color='red', label='Best-fit line')
            # plt.plot(x_fit, y_fit, label='RANSAC polynomial fit', color='red')
            plt.gca().set_aspect('equal', adjustable='box')

            # Add labels and a legend
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()

            # Save the plot to an image file
            plt.savefig(os.path.join(image_save_dir, "./drawer_line.png"))
            plt.close()

            return slope_global, button_x, button_y, button_leftmost_x, button_rightmost_x

    def compute_orientation_shelf(prompts, image_path):
        

        def compute_global_xy(x,y):
            depth = depth_image_copy[y, x] * depth_scale
            local_xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
            
            local_xyz_homogeneous = np.array([local_xyz[0], local_xyz[1], local_xyz[2], 1])
            
            global_xyz = extrinsic_matrix @ local_xyz_homogeneous
            
            assert global_xyz[3]==1           
            global_xyz = world2marker @ global_xyz
            print(global_xyz)
            return global_xyz[0], global_xyz[1]
            

        assert len(prompts)==1
        prompt = prompts[0]
        
        drawer_mask = get_mask(prompt, image_path, return_mask = True)
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        
        red_mask = (image_np[:, :, 0] > 100) & (image_np[:, :, 1] < 60) & (image_np[:, :, 2] < 60)
        combined_mask = red_mask & drawer_mask
        
        coordinates = np.argwhere(combined_mask)
        coordinates = [(coord[1], coord[0]) for coord in coordinates]
        # print(coordinates)
        # exit()
        coords_by_x = defaultdict(list)
        xs = []
        for x, y in coordinates:
            if x not in xs:
                xs.append(x)
            coords_by_x[x].append(y)
        # print(xs)
        # xs.sort()
        # print(xs)
        # assert len(xs)>=100
        # xs = xs[20:-20]
        ys = []
        for x, y_list in coords_by_x.items():
            if x in xs:
                # print(x)
                # print(y_list)
                # ys.append(random.choice(y_list))
                ys.append(int(np.median(np.array(y_list))))
        # print("xs; ys")
        # print(xs)
        # print(ys)
            
        # ys = [random.choice(y_list) for x, y_list in coords_by_x.items()]
                
        xs = np.array(xs)
        ys = np.array(ys)
        # print(xs.shape)
        # print(ys.shape)


        image = Image.open(image_path)
        image_array = np.array(image)
        plt.figure()
        fig, ax = plt.subplots()
        ax.imshow(image_array)
        
        ax.scatter(xs, ys, color='blue', label='Data points',s=2)

        plt.savefig(os.path.join(image_save_dir, f"./shelf_sampled.png"))
        plt.close()

        xs_global = []
        ys_global = []

        for i in range(xs.shape[0]):
            x_global, y_global = compute_global_xy(xs[i],ys[i])
            if x_global<0:
                continue
            xs_global.append(x_global)
            ys_global.append(y_global)
    
        xs_global = np.array(xs_global)
        ys_global = np.array(ys_global)
        # print(xs_global)
        # print("sampled points before filtering: ", xs_global.shape[0])

        xs_global_median = np.median(xs_global)
        abs_deviations = np.abs(xs_global - xs_global_median)
        # print("abs deviations: ", abs_deviations)
        threshold = 10 * np.median(abs_deviations)
        # print("filtering threshold: ", threshold)
        valid_mask = (abs_deviations <= threshold)
        # print("vaid mask: ", valid_mask)
        
        xs_global = xs_global[valid_mask]
        ys_global = ys_global[valid_mask]
        # print("sampled points after filtering: ", xs_global.shape[0])

        # slope_global, intercept_global = np.polyfit(xs_global, ys_global, 1)
   
        X = xs_global.reshape(-1, 1)
        ransac = RANSACRegressor(LinearRegression(), min_samples=20, residual_threshold=5.0)
        ransac.fit(X, ys_global)
        slope_global = ransac.estimator_.coef_[0]
        intercept_global = ransac.estimator_.intercept_

        best_fit_line_global = slope_global * xs_global + intercept_global

        plt.figure()

        plt.scatter(xs_global, ys_global, color='blue', label='Data points')
        plt.plot(xs_global, best_fit_line_global, color='red', label='Best-fit line')
        # plt.plot(x_fit, y_fit, label='RANSAC polynomial fit', color='red')
        plt.gca().set_aspect('equal', adjustable='box')

        # Add labels and a legend
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

        # Save the plot to an image file
        plt.savefig(os.path.join(image_save_dir, "./shelf_line.png"))
        plt.close()

        return slope_global



    def estimate_camera_extrinsics(image):
        def estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
            '''
            This will estimate the rvec and tvec for each of the marker corners detected by:
            corners, ids, rejectedImgPoints = detector.detectMarkers(image)
            corners - is an array of detected corners for each detected marker in the image
            marker_size - is the size of the detected markers
            mtx - is the camera matrix
            distortion - is the camera distortion matrix
            RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
            '''
            marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                    [marker_size / 2, marker_size / 2, 0],
                                    [marker_size / 2, -marker_size / 2, 0],
                                    [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
            trash = []
            rvecs = []
            tvecs = []
            
            for c in corners:
                nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
                rvecs.append(R)
                tvecs.append(t)
                trash.append(nada)
            return rvecs, tvecs, trash
        # Detect Aruco markers
        corners, ids, rejected_cand = detector.detectMarkers(image)

        # If no markers are detected, return None
        if not corners:
            return None

        # Draw detected markers (optional)
        ar.drawDetectedMarkers(image, corners, ids)
        
        # cv2.imshow("Frame with Markers", image)

        # Estimate pose of the first detected marker
        rvec, tvec, _ = estimatePoseSingleMarkers(corners[0], marker_size,
                                                    camera_matrix, distortion_coefficients)
        # print(distortion_coefficients)
        cv2.drawFrameAxes(image, camera_matrix, np.array([]), rvec[0], tvec[0], 0.1)

        # If pose estimation failed, return None
        # if not retval:
        #     return None

        return rvec, tvec

    
    aruco_dict = ar.getPredefinedDictionary(ar.DICT_6X6_250)
    detectorParams = ar.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detectorParams)
    marker_size = 0.1
    # marker_location = [0.32430097460746765-0.05, 0.1529230773448944+0.05]
    marker_location = [marker_x, marker_y, -0.02]
    
    world2marker = np.array([
        [-1,0,0,marker_location[0]],
        [0,-1,0,marker_location[1]],
        [0,0,1,marker_location[2]],
        [0,0,0,1]
    ])



    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device()


    profile = pipeline.start(config)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                            [0, intrinsics.fy, intrinsics.ppy],
                            [0, 0, 1]])
    distortion_coefficients = np.array(intrinsics.coeffs)



    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        raise NotImplementedError
        
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    # print("Depth Scale is: " , depth_scale)

    clipping_distance_in_meters = 1 
    clipping_distance = clipping_distance_in_meters / depth_scale

    align_to = rs.stream.color
    align = rs.align(align_to)


    start_time = time.time()
    stream_duration = 3  # Stream images for 3 seconds
    save_images = False
    
    
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame() #640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        

        if time.time() - start_time >= stream_duration and not save_images:
            save_images = True
            cv2.imwrite(os.path.join(image_save_dir, "rgb_image.jpg"), color_image)
            # print('rgb_image saved')
            cv2.imwrite(os.path.join(image_save_dir, "depth_image.png"), cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET))

            color_image_copy = color_image.copy()
            depth_image_copy = depth_image.copy()
            depth_colormap_copy = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET).copy()
            
            rvecs, tvecs = estimate_camera_extrinsics(color_image_copy)

            rvec = rvecs[0]
            tvec = tvecs[0]
        
            # if rvec is not None and tvec is not None:
            #     print("Rotation vector:", rvec)
            #     print("Translation vector:", tvec)

            cv2.imwrite(os.path.join(image_save_dir, "markers.jpg"), color_image_copy)
            # cv2.imshow("Image with Markers", color_image_copy)

            R, _ = cv2.Rodrigues(rvec)
            extrinsic_matrix = np.eye(4)
            extrinsic_matrix[:3,:3] = R
            extrinsic_matrix[:3,3] = tvec[:,0]
        
            # print("Camera location:", -np.dot(extrinsic_matrix[:3,:3].T, extrinsic_matrix[:3,3]))
            extrinsic_matrix = np.linalg.inv(extrinsic_matrix)

            if task == 'pill':
                global drawer_orientation
                slope, button_x, button_y, button_leftmost_x, button_rightmost_x = compute_orientation(orientation_prompts, image_path = os.path.join(image_save_dir, "rgb_image.jpg"))
            
                theta = np.arctan(slope)
                theta_degrees = np.degrees(theta)
                # print(theta_degrees)
                if theta_degrees < 0:           
                    alpha = np.pi + theta
                else:
                    alpha = theta
                drawer_orientation = np.degrees(alpha)
                # if shelf_orien_range is not None:
                #     if shelf_orientation < shelf_orien_range[0]:
                #         shelf_orientation = shelf_orien_range[0]
                #     if shelf_orientation > shelf_orien_range[1]:
                #         shelf_orientation = shelf_orien_range[1]
                print('drawer orientation:', drawer_orientation)
                # exit()
            if task == 'cereal':
                global shelf_orientation
                slope = compute_orientation_shelf(orientation_prompts, image_path = os.path.join(image_save_dir, "rgb_image.jpg"))
            
                theta = np.arctan(slope)
                theta_degrees = np.degrees(theta)
                # print(theta_degrees)
                if theta_degrees < 0:           
                    alpha = np.pi + theta
                else:
                    alpha = theta
                shelf_orientation = np.degrees(alpha)
                if shelf_orien_range is not None:
                    if shelf_orientation < shelf_orien_range[0]:
                        shelf_orientation = shelf_orien_range[0]
                    if shelf_orientation > shelf_orien_range[1]:
                        shelf_orientation = shelf_orien_range[1]
                print('shelf orientation:', shelf_orientation)
                # exit()

            image_path = os.path.join(image_save_dir, "rgb_image.jpg")

            
            known_dict = {}
            if task == 'pill':
                button_prompt = [string for string in obj_prompts if 'button' in string]
                assert len(button_prompt)==1
                button_prompt = button_prompt[0]
                known_dict[button_prompt] = [button_x, button_y, button_leftmost_x, button_rightmost_x]
            loc_dict = compute_loc(obj_prompts, image_path, known_dict)
            print("loc_dict: ", loc_dict)
            # exit()
            obj_dict = {}
            global bowl_y
            if task == 'cereal':
                for obj in obj_prompts:
                    print(obj)
                    center = loc_dict[obj]['centroid_location'] 
                    left = loc_dict[obj]['left_location']
                    right = loc_dict[obj]['right_location']  
                    center_x = center[0]
                    left_x = left[0]
                    right_x = right[0]
                    radius_estimate0 = ((left_x-center_x)+(right_x-center_x))/2
                    left_y = left[1]
                    right_y = right[1]
                    radius_estimate1 = (left_y - right_y)/2
                    if abs(radius_estimate0 - radius_estimate1)<0.005:
                        radius_estimate = (radius_estimate0+radius_estimate1)/2
                    else:
                        print("radius estimate 0:", radius_estimate0)
                        print("radius estimate 1:", radius_estimate1)
                        radius_estimate = max(radius_estimate0, radius_estimate1)

                    if obj_prompts.index(obj) == 0: #cup
                        global cup_radius
                        cup_y = center[1]
                        cup_z = center[2]
                        if fixed_radius:
                            radius_estimate = cup_radius
                        else:                    
                            cup_radius = radius_estimate
                        if latest:
                            obj_dict[processes[0]] = np.array(init_pos + [center_x+radius_estimate, center[1], lowest_grasp_point, shelf_orientation]) 
                            obj_dict[processes[1]] = np.array([center_x+radius_estimate, center[1], lowest_grasp_point, shelf_orientation])
                        elif upgraded:
                            obj_dict[processes[0]] = np.array([center_x+radius_estimate, center[1], lowest_grasp_point, shelf_orientation]) 
                            obj_dict[processes[1]] = np.array([center_x+radius_estimate, center[1], lowest_grasp_point, shelf_orientation]) 
                        else:
                            obj_dict[processes[0]] = np.array([center_x+radius_estimate, center[1], lowest_grasp_point, 90, shelf_orientation]) 
                        
                    else:
                        global bowl_radius
                        assert obj_prompts.index(obj) == len(obj_prompts)-1
                        # center_xx = (left[0] + right[0])/2 + 0.05
                        # bowl_y = (left[1]+right[1])/2          
                        # bowl_z = (left[2]+right[2])/2
                        bowl_y = center[1]
                        bowl_z = center[2]
                        
                        pour_z = bowl_z+above_bowl               
                        # global first_cup_y_larger
                        if fixed_radius:
                            radius_estimate = bowl_radius
                        else:
                            bowl_radius = radius_estimate           
                        # if cup_y > bowl_y:
                        #     # pour_y = bowl_y + (radius_estimate + cup_radius) 
                        #     pour_x = center_x + radius_estimate + math.cos(math.radians(shelf_orientation)) * (bowl_radius+cup_radius)
                        #     pour_y = bowl_y + math.sin(math.radians(shelf_orientation)) * (bowl_radius+cup_radius)
                            
                        # else:
                        #     # pour_y = bowl_y - (radius_estimate + cup_radius)
                        #     pour_x = center_x + radius_estimate - math.cos(math.radians(shelf_orientation)) * (bowl_radius+cup_radius)
                        #     pour_y = bowl_y - math.sin(math.radians(shelf_orientation)) * (bowl_radius+cup_radius) 
                        
                        if bowl_z > shelf_height:
                            orien = shelf_orientation
                            if cup_y > bowl_y:
                                pour_x = center_x + radius_estimate + math.cos(math.radians(shelf_orientation)) * (bowl_radius+cup_radius)
                                pour_y = bowl_y + math.sin(math.radians(shelf_orientation)) * (bowl_radius+cup_radius)
                                
                            else:
                                pour_x = center_x + radius_estimate - math.cos(math.radians(shelf_orientation)) * (bowl_radius+cup_radius)
                                pour_y = bowl_y - math.sin(math.radians(shelf_orientation)) * (bowl_radius+cup_radius) 
                        else:
                            orien = 90
                            if cup_y > bowl_y:
                                pour_x = center_x + radius_estimate
                                pour_y = bowl_y + (bowl_radius+cup_radius)
                                
                            else:
                                pour_x = center_x + radius_estimate
                                pour_y = bowl_y - (bowl_radius+cup_radius) 
                        if latest:
                            obj_dict[processes[1]] = np.concatenate([obj_dict[processes[1]], np.array([pour_x, pour_y, pour_z, orien])])
                            assert obj_dict[processes[0]].shape==obj_dict[processes[1]].shape
                        elif upgraded:
                            raise NotImplementedError
                            obj_dict[processes[0]] = np.concatenate([obj_dict[processes[0]], np.array([center_x+radius_estimate, pour_y, pour_z, orien])])
                            obj_dict[processes[1]] = np.concatenate([obj_dict[processes[1]], np.array([center_x+radius_estimate, pour_y, pour_z, orien])])
                            # assert obj_dict[processes[0]].shape[0]==7
                            assert (obj_dict[processes[0]]==obj_dict[processes[1]]).all()
                        else:
                            raise NotImplementedError
                            obj_dict[processes[1]] = np.array([center_x+radius_estimate, pour_y, pour_z, 90, orien]) 
                        # if loc_only:
                        #     obj_dict[processes[1]] = np.array([center_x+radius_estimate, pour_y, pour_z, 90, orien]) 
                        # else:
                        #     obj_dict[processes[1]] = np.array([center_x+radius_estimate, pour_y, pour_z, orien]) 
                        

                      
            elif task == 'pill':
                assert processes == ['open', 'pick', 'throw']
                for obj in obj_prompts:
                    print(obj)
                    center = loc_dict[obj]['centroid_location'] 
                    left = loc_dict[obj]['left_location']
                    right = loc_dict[obj]['right_location']  
                    center_x = center[0]
                    left_x = left[0]
                    right_x = right[0]
                    radius_estimate0 = ((left_x-center_x)+(right_x-center_x))/2
                    left_y = left[1]
                    right_y = right[1]
                    radius_estimate1 = (left_y - right_y)/2
                    if abs(radius_estimate0 - radius_estimate1)<0.005:
                        radius_estimate = (radius_estimate0+radius_estimate1)/2
                    else:
                        print("radius estimate 0:", radius_estimate0)
                        print("radius estimate 1:", radius_estimate1)
                        radius_estimate = max(radius_estimate0, radius_estimate1)
                    if obj_prompts.index(obj)==0: #button
                        button_height = find_closest_element(center[2], button_heights)
                        print(f"Target is the {button_heights.index(button_height)+1}/3 drawers (from bottom to top)")
                        open_x = center[0] - math.sin(math.radians(drawer_orientation)) * gripper_back
                        open_y = center[1] + math.cos(math.radians(drawer_orientation)) * gripper_back
                        
                        # print(obj_dict)
                        # exit()

                        throw_x = center[0] - math.sin(math.radians(drawer_orientation)) * (gripper_back + out_ori_button)
                        throw_y = center[1] + math.cos(math.radians(drawer_orientation)) * (gripper_back + out_ori_button)
                        assert task == 'pill'
                        obj_dict['open'] = np.array(init_pos + [open_x, open_y, button_height, 120, drawer_orientation])
                        # obj_dict['throw'] = np.array([throw_x, throw_y, button_height+above_button, drawer_orientation])

                        opened_x = center[0] - math.sin(math.radians(drawer_orientation)) * 0.12
                        opened_y = center[1] + math.cos(math.radians(drawer_orientation)) * 0.12


                        
                    else: 
                        
                        grasp_z = 0.058
                        obj_dict['pick'] = np.array([opened_x, opened_y, button_height, 120, drawer_orientation, center_x+radius_estimate, center[1], grasp_z, 165, drawer_orientation]) 
                        obj_dict['throw'] = np.array([center_x+radius_estimate, center[1], grasp_z, 165, drawer_orientation, throw_x, throw_y, button_height+above_button, 165, drawer_orientation])

            else:
                raise NotImplementedError
            
            pipeline.stop()
            cv2.destroyAllWindows()
            return obj_dict
            
        if save_images:
            raise NotImplementedError

def get_objects_preset(task):
    location_index = int(input("location index: "))
    if 'pill' in ckpt_path:
        task = 'pill'
        open_targets = [
            [],
            [],
            [],
            []
                
        ]
        pick_targets = [
            [],
            [],
            [],
            []
                
        ]
        throw_targets = [
            [],
            [],
            [], 
            []    
        ]
    elif 'cereal' in ckpt_path:
        task = 'cereal'
        pick_targets = [
            [], 
            [], 
            [], 
            [],           
        ]
        pour_targets = [
            [],  
            [],
            [],
            [],        
        ]
        relative_locations = []
        relative_location = relative_locations[location_index]




    obj_dict = {}
    if task == 'ceresl':
        global shelf_orientation
        shelf_orientation = pick_targets[location_index][-1]
        obj_dict[processes[0]] = np.concatenate([np.array(init_pos), pick_targets[location_index]])
        obj_dict[processes[1]] = np.concatenate([pick_targets[location_index], pour_targets[location_index]])
        assert obj_dict[processes[0]].shape==obj_dict[processes[1]].shape
                

    elif task == 'pill':
        global drawer_orientation
        drawer_orientation = open_targets[location_index][-1]
        obj_dict['open'] = np.concatenate([np.array(init_pos), open_targets[location_index]])
        obj_dict['pick'] = np.concatenate([open_targets[location_index], pick_targets[location_index]])
        obj_dict['throw'] = np.concatenate([pick_targets[location_index], throw_targets[location_index]])
    else:
        raise NotImplementedError            
    return obj_dict
            


