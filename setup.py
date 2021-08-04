import os
import subprocess


def main():
	# edge-connect configuration
	subprocess.call(['git', 'clone', 'https://github.com/knazeri/edge-connect.git'])
	subprocess.call(['mv', 'edge-connect', 'edgeconnect'])

	# you should manually download pre-trained model through google drive: https://drive.google.com/drive/folders/1KyXz4W4SAvfsGh3NJ7XgdOv5t46o-8aa
	# os.makedirs('./edge-connect/checkpoints', exist_ok=True)

	# write __init__.py on edgeconnect
	# with open('./edgeconnect/__init__.py', 'wb'):
	# 	# does nothing
	# 	pass




if __name__ == '__main__':
	main()