# Select relevant inputs, 0th, 2th, 4th, 6th

images = ["image 1", "image 2", "image 3", "image 4", "image 5", "image 6", "image 7"]
imgpaths = ["imgpath 1", "imgpath 2", "imgpath 3", "imgpath 4", "imgpath 5", "imgpath 6", "imgpath 7"]
self_input = "1357"
self_ground_truth = "246"
inputs = [int(e) - 1 for e in list(self_input)]  # inputs = [0,2,4,6]
gt_id = [int(e) - 1 for e in list(self_ground_truth)]  # gt_id = [1,3,5]

images = [images[i] for i in inputs]
imgpaths = [imgpaths[i] for i in inputs]
gt = [images[i] for i in gt_id]
gtpaths = [imgpaths[i] for i in gt_id]
print("inputs = ", inputs)
print("images = ", images)
print("imgpaths = ", imgpaths)
print("gt = ", gt)