import sys
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import listdir, path
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
import dlib, audio

detector = dlib.get_frontal_face_detector()
	
def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

def calcMaxArea(rects):
	max_cords = (-1,-1,-1,-1)
	max_area = 0
	max_rect = None
	for i in range(len(rects)):
		cur_rect = rects[i]
		(x,y,w,h) = rect_to_bb(cur_rect)
		if w*h > max_area:
			max_area = w*h
			max_cords = (x,y,w,h)
			max_rect = cur_rect	
	return max_cords, max_rect

def face_detect(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)
	(x, y, w, h), max_rect = calcMaxArea(rects)
	if x == -1:
		return None, False
	faceAligned = image[y:y+h, x:x+w]
	if 0 in faceAligned.shape: return None, False
	return faceAligned, True

step_size_in_ms = 40
window_size = 350
mfcc_chunk_size = window_size // 10
mfcc_step_size = 4
fps = 25
video_step_size_in_ms = mfcc_step_size * 10 # for 25 fps video
sr = 16000

template = 'ffmpeg -loglevel panic -y -i {} -ar {} {}'

def process_video_file(vfile, args, split):
	video_stream = cv2.VideoCapture(vfile)
	frames = []
	while 1:
		still_reading, frame = video_stream.read()
		if not still_reading:
			video_stream.release()
			break
		frames.append(frame)
	mid_frames = []
	ss = 0.
	es = (ss + (window_size / 1000.))

	while int(es * fps) <= len(frames):
		mid_second = (ss + es) / 2.
		mid_frames.append(frames[int(mid_second * fps)])

		ss += (video_step_size_in_ms / 1000.)
		es = (ss + (window_size / 1000.))


	dst_subdir = path.join(vfile.split('/')[-2], vfile.split('/')[-1].split('.')[0])
	fulldir = path.join(args.final_data_root, split, dst_subdir)
	os.makedirs(fulldir, exist_ok=True)
	wavpath = path.join(fulldir, 'audio.wav')

	command = template.format(vfile, sr, wavpath)
	subprocess.call(command, shell=True)

	specpath = path.join(fulldir, 'mels.npz')

	if path.isfile(wavpath):
		wav = audio.load_wav(wavpath, sr)

		spec = audio.melspectrogram(wav)
		np.savez_compressed(specpath, spec=spec)
	else:
		return

	for i, f in enumerate(mid_frames):
		face, valid_frame = face_detect(f)

		if not valid_frame:
			continue
		
		resized_face = cv2.resize(face, (args.img_size, args.img_size))
		
		cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), resized_face)
	
def mp_handler(job):
	vfile, args, split = job
	try:
		process_video_file(vfile, args, split)
	except:
		traceback.print_exc()
		
def dump_split(args):
	print('Started processing for {} with {} CPU cores'.format(args.split, args.num_workers))

	filelist = [path.join(args.videos_data_root, ('pretrain' if args.split == 'pretrain' else 'main'), 
				'{}.mp4'.format(line.strip())) \
				for line in open(path.join(args.filelists, '{}.txt'.format(args.split))).readlines()]
	
	jobs = [(vfile, args, ('pretrain' if args.split == 'pretrain' else 'main')) for vfile in filelist]
	p = ThreadPoolExecutor(args.num_workers)
	futures = [p.submit(mp_handler, j) for j in jobs]
	_ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]



if __name__ == '__main__':
	dump_split(args)
