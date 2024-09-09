from random import randint
import numpy as np
import os
from munch import DefaultMunch
import subprocess

def generation_pipeline(luma, radius, amount, qf):
    pipeline = dict(
    directional_pyramid_denoising={
        'enabled': True,
        'luma': luma
    }, sharpening={
        'enabled': True,
        'radius': radius,
        'amount': amount
    }, jpeg={
        'qf': qf
    })
    return pipeline
def yaml_to_rt_pipeline(pipeline, output_conf_file, ref_file):
    '''
    Function that takes the output of generation_pipeline and from the ref.pp3 file create a new .pp3 file
    '''
    values = dict()
    values["[Directional Pyramid Denoising]"] = dict()
    values["[Sharpening]"] = dict()
    values["[Directional Pyramid Denoising]"]["Enabled"] = str(pipeline.directional_pyramid_denoising.enabled).lower() if pipeline.directional_pyramid_denoising.enabled else "false"
    values["[Directional Pyramid Denoising]"]["Luma"] = str(pipeline.directional_pyramid_denoising.luma).lower() if pipeline.directional_pyramid_denoising.luma else "0"
    values["[Sharpening]"]["Enabled"] = str(pipeline.sharpening.enabled).lower() if pipeline.sharpening.enabled else "false"
    values["[Sharpening]"]["Radius"] = str(pipeline.sharpening.radius).lower() if pipeline.sharpening.radius else "0.3"
    values["[Sharpening]"]["Amount"] = str(pipeline.sharpening.amount).lower() if pipeline.sharpening.amount else "0"

    watch = False
    key = None
    try:
        with open(output_conf_file, "w") as fw:
            try:
                with open(ref_file, "r") as fr:
                    line = fr.readline()
                    while line:
                        if line[0] == '[':
                            if line[:-1] in values:
                                watch = True
                                key = line[:-1]
                            else:
                                watch = False
                                key = None
                        if watch:
                            tmp = line.split('=')[0]
                            if tmp in values[key]:
                                line = tmp + '=' + values[key][tmp] + '\n'
                        fw.write(line)
                        line = fr.readline()
            except Exception as e:
                print("Failed to open the file:", ref_file, "The error is:", e)
    except Exception as e:
        print("Failed to open the file:", output_conf_file, "The error is:", e)

    print('Pipeline created')

def develop_tif_pictures(input_path, output_folder, pp3_path, qf):
    for img in os.listdir(input_path):
        if img.startswith('.') :
            continue
        else :
            img_path = os.path.join(input_path, img)
            tif_image_name = os.path.join(output_folder, os.path.splitext(img)[0] + ".tif")

            # RawTherapee development command
            development_cmd = [
                "rawtherapee-cli", "-Y", "-o", tif_image_name, "-t16", "-p", pp3_path, "-c", img_path
            ]
            try:
                result = subprocess.run(development_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print("Error occurred:", e)
                print(e.stderr.decode())

            # Final JPEG compression with ImageMagick
            jpeg_output_path = os.path.join(output_folder, os.path.splitext(img)[0] + ".jpg")
            jpeg_compression_cmd = [
                "convert", "-define", "jpeg:optimize-coding=false", "-quality", str(qf),
                tif_image_name, jpeg_output_path
            ]
            try:
                result = subprocess.run(jpeg_compression_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print("Error occurred:", e)
                print(e.stderr.decode())

            os.remove(tif_image_name)

def create_output_directories(origin_folder, file_name_2, folder):
    new_dir = os.path.join(origin_folder, file_name_2, folder)
    if os.path.exists(new_dir):
        pass
    else :
        os.makedirs(new_dir)

    print(f'Successfully created {new_dir}')
    return new_dir

for luma in range(1,62,15):
    for radius in np.arange(0.01,1.52,0.5):
        amount = 300
        # Sets a random integer for qf
        qf = 70
        # Generate a dictionary with the values for the transformation
        pipeline = generation_pipeline(luma, radius, amount, qf)
        pipeline = DefaultMunch.fromDict(pipeline)
        print(f'luma : {luma} - radius : {radius} - amount {amount}- qf : {qf}')

        # Create the name of the pp3 file + the output path in a specified folder
        file_name_pp3 = """luma-{0}-radius-{1}-amount-{2}-qf-{3}.pp3""".format(luma, radius, amount, qf)
        output_pp3 = os.path.join("fichiers_pp3", file_name_pp3)
        if os.path.exists(output_pp3) == False :
            print(output_pp3 + " does not exist")
        else :
            print(output_pp3 + " exist")

        # Create the pp3 file
        yaml_to_rt_pipeline(pipeline, output_pp3, 'fichiers_pp3/ref.pp3')

        # Create the 2 output paths for the new transformed images
        file_name_2 = """luma-{0}-radius-{1}-amount-{2}-qf-{3}""".format(luma, radius, amount, qf)
        output_true = create_output_directories('forgery_toybase2/evaluation', file_name_2, 'true')
        output_fake = create_output_directories('forgery_toybase2/evaluation', file_name_2, 'fake')

        #Apply the transformations to the images
        develop_tif_pictures('forgery_toybase2/evaluation/Sans_Compression/true', output_true,output_pp3, qf)
        develop_tif_pictures('forgery_toybase2/evaluation/Sans_Compression/fake', output_fake,output_pp3, qf)
