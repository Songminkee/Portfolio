from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

def load_kocca_param_demo(path):
    num, side = path.shape
    l_k = np.zeros((num, 3 * 3), dtype=np.float32)
    l_quar = np.zeros((num,4),dtype=np.float32)
    l_c = np.zeros((num, 3), dtype=np.float32)
    r_k = np.zeros((num, 3 * 3), dtype=np.float32)
    r_quar = np.zeros((num, 4), dtype=np.float32)
    r_c = np.zeros((num, 3), dtype=np.float32)
    for i in range(num):
        f_l = open(path[i][0], 'r')
        line_l = f_l.readlines()
        l_k[i] = (np.array(line_l[0].replace('\n', '').split(' '))[:3 * 3]).astype(np.float32)
        l_quar[i] = (np.array(line_l[1].replace('\n', '').split(' '))[:4]).astype(np.float32)
        l_c[i] = (np.array(line_l[2].replace('\n', '').split(' '))[:3]).astype(np.float32)

        f_r = open(path[i][1], 'r')
        line_r = f_r.readlines()
        r_k[i] = (np.array(line_r[0].replace('\n', '').split(' '))[:3 * 3]).astype(np.float32)
        r_quar[i] = (np.array(line_r[1].replace('\n', '').split(' '))[:4]).astype(np.float32)
        r_c = (np.array(line_r[2].replace('\n', '').split(' '))[:3]).astype(np.float32)

        f_l.close()
        f_r.close()

    return l_k, l_quar, l_c, r_k, r_quar, r_c

def quar_to_rot(quar):
    xx=1-(2*(tf.pow(quar[:,:,2],2)+tf.pow(quar[:,:,3],2)))
    xy=2*(quar[:,:,1]*quar[:,:,2]-quar[:,:,0]*quar[:,:,3])
    xz=2*(quar[:,:,0]*quar[:,:,2]+quar[:,:,1]*quar[:,:,3])
    yx=2*(quar[:,:,1]*quar[:,:,2]+quar[:,:,0]*quar[:,:,3])
    yy=1-(2*(tf.pow(quar[:,:,1],2)+tf.pow(quar[:,:,3],2)))
    yz=2*(quar[:,:,2]*quar[:,:,3]-quar[:,:,0]*quar[:,:,1])
    zx=2*(quar[:,:,1]*quar[:,:,3]-quar[:,:,0]*quar[:,:,2])
    zy=2*(quar[:,:,0]*quar[:,:,1]-quar[:,:,2]*quar[:,:,3])
    zz=1-(2*(tf.pow(quar[:,:,1],2)+tf.pow(quar[:,:,2],2)))

    rot_x = tf.expand_dims(tf.concat([xx, xy, xz], 1),1)
    rot_y = tf.expand_dims(tf.concat([yx, yy, yz], 1),1)
    rot_z = tf.expand_dims(tf.concat([zx, zy, zz], 1),1)

    rot = tf.concat([rot_x,rot_y,rot_z],1)

    return rot

def quar_interpolation(left,right,t):
    left_t = tf.transpose(left,[0,2,1])
    theta = tf.acos(tf.matmul(right,left_t))
    start = left
    end = right
    target_quar = (start*tf.sin((1-t)*theta)+end*tf.sin(t*theta))/tf.sin(theta)

    return target_quar

def cal_t(r,c):
    t= -1*tf.matmul(r,c)
    return t

def meshgrid(num, height, width):
  x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                  tf.transpose(tf.expand_dims(
                      tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
  y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                  tf.ones(shape=tf.stack([1, width])))
  x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
  y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
  ones = tf.ones_like(x_t)
  coords = tf.stack([x_t, y_t, ones], axis=0)
  coords = tf.tile(tf.expand_dims(coords, 0), [num, 1, 1, 1])
  return coords


def pixel2cam(depth, pixel_coords, target_intrinsics):
    num, height, width,_ = depth.get_shape().as_list()
    depth = tf.reshape(depth,[num,1, -1])
    pixel_coords = tf.reshape(pixel_coords, [num,3, -1])
    cam_coords = tf.matmul(tf.matrix_inverse(target_intrinsics),
                           pixel_coords * depth)
    cam_coords = tf.reshape(cam_coords, [num,-1, height, width])
    return cam_coords

def cam2world(cam_coords, target_pose):
  num,axis,height, width = cam_coords.get_shape().as_list()
  rotation = tf.slice(target_pose, [0, 0, 0], [num, 3, 3])
  translation = tf.slice(target_pose, [0, 0, 3], [num, 3, 1])

  cam_coords = tf.reshape(cam_coords, [num,axis, -1])
  cam_m = cam_coords - translation
  world_coords = tf.matmul(tf.matrix_inverse(rotation), cam_m) ## camera_coordin = k^-1(3x3) * p(3x1) * z(1x1) = (3x1)

  ones = tf.ones([num,1, height*width])
  world_coords = tf.concat([world_coords, ones], axis=1) # 3x1 -> 4x1
  world_coords = tf.reshape(world_coords, [num,-1, height, width])

  return world_coords

def projective_depth_forward_warp(depth,target_pose,target_intrinsics,src_pose,src_intrinsics):
    num, height, width, _ = depth.get_shape().as_list()

    target_intrinsics = tf.concat(
        [[tf.expand_dims(
            tf.concat([target_intrinsics[i][0] * width, target_intrinsics[i][1] * height, target_intrinsics[i][2]], 0),
            0)] for i in range(num)],
        0)
    target_intrinsics = tf.reshape(target_intrinsics, [num, 3, 3])
    src_intrinsics = tf.concat(
        [[tf.expand_dims(
            tf.concat([src_intrinsics[i][0] * width, src_intrinsics[i][1] * height, src_intrinsics[i][2]], 0), 0)] for i
         in range(num)],
        0)
    src_intrinsics = tf.reshape(src_intrinsics, [num, 3, 3])

    pixel_coords = meshgrid(num, height, width)

    cam_coords = pixel2cam(depth, pixel_coords, target_intrinsics)  ## pixel_coord to camera_coordin

    world_coords = cam2world(cam_coords, target_pose)

    proj_src = tf.matmul(src_intrinsics, src_pose)

    depth_warp,src_pixel_coords=depth_warping_and_world2pixel(world_coords,proj_src)

    mask,over_coords,warped_depth = depth_forward_sampler(depth_warp,src_pixel_coords)


    return over_coords,warped_depth

def depth_synthesis(warped_depth1,warped_depth2,mask1,mask2,MAX):
    warped_depth1=tf.stop_gradient(warped_depth1)
    warped_depth2 = tf.stop_gradient(warped_depth2)
    mask1 = tf.stop_gradient(mask1)
    mask2 = tf.stop_gradient(mask2)
    depth_mask1 = tf.math.logical_and(tf.math.logical_not(mask1), tf.less_equal(warped_depth1, warped_depth2+tf.cast(mask2,'float32')*MAX))
    depth_mask2 = tf.math.logical_and(tf.math.logical_not(mask2), tf.less_equal(warped_depth2, warped_depth1+tf.cast(mask1,'float32')*MAX))
    common_mask = tf.math.logical_and(depth_mask1,depth_mask2)

    return depth_mask1,depth_mask2,common_mask

def depth_forward_sampler(imgs, coords):
    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
            ])), 1), [1, 0])
        rep = tf.cast(rep, 'float32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    with tf.name_scope('image_sampling'):
        coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
        inp_size = imgs.get_shape()
        coord_size = coords.get_shape()
        out_size = coords.get_shape().as_list()
        out_size[3] = imgs.get_shape().as_list()[3]

        y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
        x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
        zero = tf.zeros([1], dtype='float32')

        x_safe = tf.clip_by_value(tf.floor(coords_x + 0.5), zero, x_max)
        y_safe = tf.clip_by_value(tf.floor(coords_y + 0.5), zero, y_max)

        out_coords = tf.math.logical_or(tf.not_equal(tf.floor(coords_x + 0.5), x_safe),  tf.not_equal(tf.floor(coords_y + 0.5), y_safe))

        imgs = imgs + tf.cast(out_coords, 'float32') * 101

        dim2 = tf.cast(inp_size[2], 'float32')
        dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
        base = tf.reshape(_repeat(tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
                coord_size[1] * coord_size[2]),
            [out_size[0], out_size[1], out_size[2], 1])
        base_y = base + y_safe * dim2
        idx = tf.reshape(x_safe + base_y, [-1, 1])

        seg_min = tf.unsorted_segment_min(tf.reshape(imgs, [-1]), tf.cast(tf.reshape(idx, [-1]), 'int32'),
                                          num_segments=inp_size[0] * inp_size[1] * inp_size[2])
        seg_min_resize= tf.reshape(seg_min, out_size)

        im = seg_min_resize * tf.cast(tf.less(seg_min_resize, 101), 'float32')

        hole_mask = tf.reshape(scatter_mask_zero(idx), tf.shape(coords_x))
        return tf.cast(hole_mask, 'bool'),tf.cast(hole_mask,'bool'), im

def bilinear_sampler(imgs, coords):
  def _repeat(x, n_repeats):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.stack([
            n_repeats,
        ])), 1), [1, 0])
    rep = tf.cast(rep, 'float32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])

  with tf.name_scope('image_sampling'):
    coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
    inp_size = imgs.get_shape()
    coord_size = coords.get_shape()
    out_size = coords.get_shape().as_list()
    out_size[3] = imgs.get_shape().as_list()[3]

    coords_x = tf.cast(coords_x, 'float32')
    coords_y = tf.cast(coords_y, 'float32')

    x0 = tf.floor(coords_x)
    x1 = x0 + 1
    y0 = tf.floor(coords_y)
    y1 = y0 + 1

    y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
    x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
    zero = tf.zeros([1], dtype='float32')

    x_safe = tf.clip_by_value(coords_x,zero,x_max)
    y_safe = tf.clip_by_value(coords_y,zero,y_max)
    coords_safe = tf.concat([x_safe,y_safe],-1)

    x0_safe = tf.clip_by_value(x0, zero, x_max)
    y0_safe = tf.clip_by_value(y0, zero, y_max)
    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)

    wt_x0 = x1_safe - coords_x
    wt_x1 = coords_x - x0_safe
    wt_y0 = y1_safe - coords_y
    wt_y1 = coords_y - y0_safe

    dim2 = tf.cast(inp_size[2], 'float32')
    dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
    base = tf.reshape(_repeat(tf.cast(tf.range(coord_size[0]), 'float32') * dim1, coord_size[1] * coord_size[2]),
        [out_size[0], out_size[1], out_size[2], 1])

    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    idx00 = tf.reshape(x0_safe + base_y0, [-1])
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
    imgs_flat = tf.cast(imgs_flat, 'float32')
    im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
    im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
    im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
    im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = tf.add_n([
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
    ])
    return output,coords_safe

def scatter_mask_zero(idx_):
    idx = tf.cast(idx_, 'int32')
    ones = tf.ones_like(idx,'float32')
    scat = tf.scatter_nd(idx, ones, tf.shape(idx))
    mask = tf.where(tf.equal(scat,ones-1),ones,ones-1)

    return mask

def world2pixel(world_coords,proj_src):
    num,axis,height,width = world_coords.get_shape().as_list()
    world_coords = tf.reshape(world_coords,[num,4,-1])

    unnormalized_pixel_coords = tf.matmul(proj_src,world_coords)
    x_u = tf.slice(unnormalized_pixel_coords, [0,0, 0],
                   [-1,1, -1])
    y_u = tf.slice(unnormalized_pixel_coords, [0,1, 0], [-1,1, -1])
    z_u = tf.slice(unnormalized_pixel_coords, [0,2, 0], [-1,1, -1])

    x_n = (x_u / (z_u + 1e-10))
    y_n = (y_u / (z_u + 1e-10))

    src_pixel_coords = tf.concat([x_n, y_n], axis=1)
    src_pixel_coords = tf.reshape(src_pixel_coords, [num,2, height, width])
    return tf.transpose(src_pixel_coords, perm=[0, 2, 3, 1])

def forward_sampler(imgs, coords):
    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
            ])), 1), [1, 0])
        rep = tf.cast(rep, 'float32')
        x = tf.matmul(tf.reshape(x, (-1, 1)),
                      rep)
        return tf.reshape(x, [-1])
    with tf.name_scope('image_sampling'):
        coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
        inp_size = imgs.get_shape()
        coord_size = coords.get_shape()
        out_size = coords.get_shape().as_list()
        out_size[3] = imgs.get_shape().as_list()[3]

        y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
        x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
        zero = tf.zeros([1], dtype='float32')

        x_safe = tf.clip_by_value(tf.floor(coords_x+0.5), zero, x_max)
        y_safe = tf.clip_by_value(tf.floor(coords_y+0.5), zero, y_max)

        dim2 = tf.cast(inp_size[2], 'float32')
        dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
        base = tf.reshape(
            _repeat(
                tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
                coord_size[1] * coord_size[2]),
            [out_size[0], out_size[1], out_size[2], 1])
        base_y = base + y_safe * dim2

        idx = tf.reshape(x_safe + base_y, [-1, 1])

        imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
        imgs_flat = tf.cast(imgs_flat, 'float32')

        im = tf.reshape(tf.scatter_nd(tf.cast(idx, 'int32'), imgs_flat, tf.shape(imgs_flat)),
                          out_size)

        hole_mask = tf.reshape(scatter_mask_zero(idx),tf.shape(coords_x))

        return tf.clip_by_value(im, zero, tf.ones_like(zero)),hole_mask


def bilinear_sampler_forward(imgs, coords):

    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
            ])), 1), [1, 0])
        rep = tf.cast(rep, 'float32')
        x = tf.matmul(tf.reshape(x, (-1, 1)),
                      rep)
        return tf.reshape(x, [-1])

    with tf.name_scope('image_sampling'):
        coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
        inp_size = imgs.get_shape()
        coord_size = coords.get_shape()
        out_size = coords.get_shape().as_list()
        out_size[3] = imgs.get_shape().as_list()[3]

        x0 = tf.floor(coords_x)
        x1 = x0 + 1
        y0 = tf.floor(coords_y)
        y1 = y0 + 1

        y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
        x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
        zero = tf.zeros([1], dtype='float32')

        x0_safe = tf.clip_by_value(x0, zero, x_max)
        y0_safe = tf.clip_by_value(y0, zero, y_max)
        x1_safe = tf.clip_by_value(x1, zero, x_max)
        y1_safe = tf.clip_by_value(y1, zero, y_max)

        wt_x0 = x1_safe - coords_x
        wt_x1 = coords_x - x0_safe
        wt_y0 = y1_safe - coords_y
        wt_y1 = coords_y - y0_safe

        dim2 = tf.cast(inp_size[2], 'float32')
        dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
        base = tf.reshape(
            _repeat(tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
            coord_size[1] * coord_size[2]),
            [out_size[0], out_size[1], out_size[2], 1])

        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = tf.reshape(x0_safe + base_y0, [-1,1])
        idx01 = tf.reshape(x0_safe + base_y1 , [-1,1])
        idx10 = tf.reshape(x1_safe + base_y0 , [-1,1])
        idx11 = tf.reshape(x1_safe + base_y1 , [-1,1])

        imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
        imgs_flat = tf.cast(imgs_flat, 'float32')

        im00 = tf.reshape(tf.scatter_nd(tf.cast(idx00, 'int32'), imgs_flat, tf.shape(imgs_flat)) ,out_size)
        im01 = tf.reshape(tf.scatter_nd(tf.cast(idx01, 'int32'), imgs_flat, tf.shape(imgs_flat)) , out_size)
        im10 = tf.reshape(tf.scatter_nd(tf.cast(idx10, 'int32'), imgs_flat, tf.shape(imgs_flat)) , out_size)
        im11 = tf.reshape(tf.scatter_nd(tf.cast(idx11, 'int32'), imgs_flat, tf.shape(imgs_flat)) , out_size)

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        output = tf.add_n([
            w00 * im00, w01 * im01,
            w10 * im10, w11 * im11
        ])
        return tf.clip_by_value(output,zero,tf.ones_like(zero))

def projective_forward_warp(img,depth,target_pose,target_intrinsics,src_pose,src_intrinsics):

    num, height, width, _ = img.get_shape().as_list()
    target_intrinsics = tf.concat(
        [[tf.expand_dims(
            tf.concat([target_intrinsics[i][0] * width, target_intrinsics[i][1] * height, target_intrinsics[i][2]], 0),
            0)] for i in range(num)],
        0)
    target_intrinsics = tf.reshape(target_intrinsics, [num, 3, 3])
    src_intrinsics = tf.concat(
        [[tf.expand_dims(
            tf.concat([src_intrinsics[i][0] * width, src_intrinsics[i][1] * height, src_intrinsics[i][2]], 0), 0)] for i
         in range(num)],
        0)
    src_intrinsics = tf.reshape(src_intrinsics, [num, 3, 3])

    pixel_coords = meshgrid(num, height, width)

    cam_coords = pixel2cam(depth, pixel_coords, target_intrinsics)

    world_coords = cam2world(cam_coords, target_pose)

    proj_src = tf.matmul(src_intrinsics, src_pose)

    src_pixel_coords = world2pixel(world_coords, proj_src)

    output_img,mask = forward_sampler(img,src_pixel_coords)

    return output_img,mask,src_pixel_coords


def synthesis_reconstruction(mask1, mask2, common_mask, Im_l, Im_r, refer_pose=None, pose1=None, pose2=None,inpaint=False):
    num, _, _ = refer_pose.get_shape().as_list()
    refer_rotation = tf.slice(refer_pose, [0, 0, 0], [num, 3, 3])
    refer_translation = tf.slice(refer_pose, [0, 0, 3], [num, 3, 1])
    refer_c = -1 * tf.matmul(tf.matrix_inverse(refer_rotation), refer_translation)

    rotation_1 = tf.slice(pose1, [0, 0, 0], [num, 3, 3])
    translation_1 = tf.slice(pose1, [0, 0, 3], [num, 3, 1])
    c1 = -1 * tf.matmul(tf.matrix_inverse(rotation_1), translation_1)
    d1 = tf.norm(refer_c - c1, axis=1)


    rotation_2 = tf.slice(pose2, [0, 0, 0], [num, 3, 3])
    translation_2 = tf.slice(pose2, [0, 0, 3], [num, 3, 1])
    c2 = -1 * tf.matmul(tf.matrix_inverse(rotation_2), translation_2)
    d2 = tf.norm(refer_c - c2, axis=1)

    w1 = tf.reshape(d2 / (d1 + d2), [num, 1, 1, 1])
    w2 = tf.reshape(d1 / (d1 + d2), [num, 1, 1, 1])

    not_common_mask = tf.math.logical_not(common_mask)

    l_common = Im_l* tf.cast(common_mask, 'float32')*w1
    r_common = Im_r* tf.cast(common_mask,'float32')*w2
    common_im = l_common+r_common
    common_hole = tf.cast(tf.math.logical_not(tf.math.logical_or(mask1, mask2)), 'float32')

    synthesis = tf.cast(tf.math.logical_and(mask1,not_common_mask),'float32')*Im_l+Im_r*tf.cast(tf.math.logical_and(mask2,not_common_mask),'float32')+common_im#+common_hole*(Im_l+Im_r)*0.5


    if inpaint:
        pool = synthesis+tf.nn.avg_pool(synthesis,[1, 8, 8, 1], [1, 1, 1, 1], 'SAME') * tf.cast(common_hole,'float32')
        pool = pool*(1- tf.cast(common_hole,'float32')) +tf.nn.avg_pool(pool,[1, 8, 8, 1], [1, 1, 1, 1], 'SAME') * tf.cast(common_hole,'float32')
        pool = pool*(1- tf.cast(common_hole,'float32')) +tf.nn.avg_pool(pool,[1, 8, 8, 1], [1, 1, 1, 1], 'SAME') * tf.cast(common_hole,'float32')
        pool = pool * (1 - tf.cast(common_hole, 'float32')) + tf.nn.avg_pool(pool, [1, 8, 8, 1], [1, 1, 1, 1],
                                                                         'SAME') * tf.cast(common_hole, 'float32')
        pool = pool * (1 - tf.cast(common_hole, 'float32')) + tf.nn.avg_pool(pool, [1, 8, 8, 1], [1, 1, 1, 1],
                                                                         'SAME') * tf.cast(common_hole, 'float32')
        pool = pool * (1 - tf.cast(common_hole, 'float32')) + tf.nn.avg_pool(pool, [1, 8, 8, 1], [1, 1, 1, 1],
                                                                             'SAME') * tf.cast(common_hole, 'float32')
        pool = pool * (1 - tf.cast(common_hole, 'float32')) + tf.nn.avg_pool(pool, [1, 8, 8, 1], [1, 1, 1, 1],
                                                                             'SAME') * tf.cast(common_hole, 'float32')
        pool = pool * (1 - tf.cast(common_hole, 'float32')) + tf.nn.avg_pool(pool, [1, 8, 8, 1], [1, 1, 1, 1],
                                                                             'SAME') * tf.cast(common_hole, 'float32')
        synthesis = pool * (1 - tf.cast(common_hole, 'float32')) + tf.nn.avg_pool(pool, [1, 8, 8, 1], [1, 1, 1, 1],
                                                                             'SAME') * tf.cast(common_hole, 'float32')

    return synthesis,common_hole

def projective_inverse_warp(img,depth,target_pose,target_intrinsics,src_pose,src_intrinsics):
    num, height, width, _ = img.get_shape().as_list()

    target_intrinsics=tf.concat(
        [[tf.expand_dims(tf.concat([target_intrinsics[i][0]*width, target_intrinsics[i][1]*height, target_intrinsics[i][2]], 0),0)] for i in range(num)],
        0)
    target_intrinsics=tf.reshape(target_intrinsics,[num,3,3])
    src_intrinsics =tf.concat(
        [[tf.expand_dims(tf.concat([src_intrinsics[i][0]*width, src_intrinsics[i][1]*height, src_intrinsics[i][2]], 0),0)] for i in range(num)],
        0)
    src_intrinsics=tf.reshape(src_intrinsics,[num,3,3])

    pixel_coords = meshgrid(num,height,width)

    cam_coords = pixel2cam(depth, pixel_coords, target_intrinsics)
    world_coords = cam2world(cam_coords, target_pose)
    proj_src = tf.matmul(src_intrinsics, src_pose)
    src_pixel_coords = world2pixel(world_coords, proj_src)
    output_img,coords = bilinear_sampler(img, src_pixel_coords)

    coord_mask = tf.greater(tf.abs(src_pixel_coords-coords),tf.zeros_like(src_pixel_coords))
    coord_mask = tf.math.logical_or(coord_mask[..., 0], coord_mask[..., 1])
    return output_img,tf.expand_dims(coord_mask,-1)


def depth_warping_and_world2pixel(world_coords,proj):
    num, axis, height, width = world_coords.get_shape().as_list()
    world_coords = tf.reshape(world_coords, [num, 4, -1])

    unnormalized_pixel_coords = tf.matmul(proj, world_coords)
    x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0],
                   [-1, 1, -1])
    y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
    z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])

    n_depth = tf.reshape(z_u,[num,height,width,1])
    x_n = (x_u / (z_u + 1e-10))
    y_n = (y_u / (z_u + 1e-10))

    src_pixel_coords = tf.concat([x_n, y_n], axis=1)
    src_pixel_coords = tf.reshape(src_pixel_coords, [num, 2, height, width])
    src_pixel_coords = tf.transpose(src_pixel_coords, perm=[0, 2, 3, 1])

    return n_depth,src_pixel_coords

