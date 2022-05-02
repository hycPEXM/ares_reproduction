import subprocess
# import pathlib
from pathlib import Path
import os
import sys

key_list=['rms', 'rms_stem', 'score']

def get_PDBfile_list(pdb_path):
    if os.path.isdir(pdb_path):
        find_cmd = r"find {:} -regex '.*\.{:}'".format(pdb_path, "pdb$")
        out = subprocess.Popen(
            find_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=os.getcwd(), shell=True)
        (stdout, stderr) = out.communicate()
        name_list = stdout.decode().split()
        name_list.sort()
        return [Path(x).absolute() for x in name_list]

def build_sc_from_pdb(pdb_path, sc_path):
    pdb_file_list=get_PDBfile_list(pdb_path)
    for i in pdb_file_list:           #直接i也行，不用i.as_posix()
        last30 = subprocess.Popen("tail -n 30 {}".format(i.as_posix()), \
                              stdout=subprocess.PIPE, shell=True)
        sc_filename=str(i.stem)+'.sc'
        fo=open(os.path.join(sc_path, sc_filename),"w")
        # fo.write("SCORE: \n")
        # fo.write("SCORE: \n")
        line1=['SCORE:']
        line2=['SCORE:']
        for line in last30.stdout.readlines():
            # print(line.decode())
            if line:
                line=line.decode('utf-8')
                line=line.strip()
                line_list=line.split()
                if len(line_list)>=2:
                    line_list0=line_list[0]
                    line_list1=line_list[1]
                    if line_list0 in key_list:
                        #print(line_list0)
                        line1.append(' '*5)
                        line2.append(' '*5)
                        len_key=len(line_list0)
                        len_value=len(line_list1)
                        if len_key>len_value:
                            line_list1=' '*int(len_key-len_value)+line_list1
                        elif len_key<len_value:
                            line_list0=' '*int(len_value-len_key)+line_list0
                        line1.append(line_list0)
                        line2.append(line_list1)
        line1.append(' '*5)
        line2.append(' '*5)
        des_str='description'
        pdb_name_str=i.name
        len_des=len(des_str)
        len_pdb_name=len(pdb_name_str)
        if(len_des>len_pdb_name):
            pdb_name_str=' '*int(len_des-len_pdb_name)+pdb_name_str
        elif(len_des<len_pdb_name):
            des_str=' '*int(len_pdb_name-len_des)+des_str
        line1.append(des_str)
        line1.append('\n')
        line2.append(pdb_name_str)
        line2.append('\n')
        fo.writelines(line1)
        fo.writelines(line2)
        fo.close()

if __name__ == "__main__":
    pdb_path=sys.argv[1]  
    sc_path=sys.argv[2]
    build_sc_from_pdb(pdb_path, sc_path)