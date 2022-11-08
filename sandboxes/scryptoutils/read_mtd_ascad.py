import tables
import numpy as np
from scryptoutils import AES, encrypt, decrypt
import binascii
from tqdm import trange

def print_hex(all_data):
    result = ''
    for x in all_data:
        result += hex(x)[2:].zfill(2)
    return result

ascad_file = tables.open_file(r'D:\AES_PT\AES_PT_masked-D1.h5', mode='r')
ascad = ascad_file.root.Attack_traces.metadata
all_data = np.concatenate((ascad[0]['plaintext'], ascad[0]['key'], ascad[0]['masks'], ascad[0]['ciphertext']), axis=0)
print (all_data[0:16])
plaintext_byte = np.array(all_data[0:16], np.uint8).tobytes()
plaintext = print_hex(all_data[0:16])
print ('Plain text:', plaintext)
print ('-------------------------------')
print (all_data[16:32])
key_byte = np.array(all_data[16:32], np.uint8).tobytes()
key = print_hex(all_data[16:32])
print ('Key:', key)
#print ([hex(x)[2:] for x in all_data[16:32]])
print ('-------------------------------')
print (all_data[32:34])
print ('Masks:', print_hex(all_data[32:34]))
#print ([hex(x)[2:] for x in all_data[32:50]])
print ('-------------------------------')
print (all_data[34:50])
ciphertext_byte = np.array(all_data[34:50], np.uint8).tobytes()
print ('Cipher text:', print_hex(all_data[34:50]))
expected_ciphertext = AES(bytes(key_byte)).encrypt_block(bytes(plaintext_byte))
print ('cipher text:', binascii.hexlify(bytearray(expected_ciphertext)).decode('ascii'), '(expected)')
print ('-------------------------------')
plaintext_result = AES(bytes(key_byte)).decrypt_block(bytes(ciphertext_byte))
print ('Plain text:', binascii.hexlify(bytearray(plaintext_result)).decode('ascii'))
print ('Plain text:', plaintext, '(expected)')

amout_bad_traces = 0
with open('AES_PT_masked-D1_attack_wrong_traces.txt', 'w') as file:

    for trace in trange(len(ascad)):
        all_data = np.concatenate((ascad[0]['plaintext'], ascad[0]['key'], ascad[0]['masks'], ascad[0]['ciphertext']), axis=0)
        numpyct = all_data[34:50]
        plaintext_byte = np.array(all_data[0:16], np.uint8).tobytes()
        expnumpyct = AES(bytes(key_byte)).encrypt_block(bytes(plaintext_byte))
        expnumpyct = np.frombuffer(expnumpyct, dtype=np.uint8)

        for i, element in enumerate(expnumpyct):
            if numpyct[i] != element:
                file.write(str(trace) + '\n')
                amout_bad_traces += 1
                break 

print ('bad traces:', amout_bad_traces)
ascad_file.close()