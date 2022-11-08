import trsfile
import numpy as np
from scryptoutils import AES, encrypt, decrypt
import binascii
from tqdm import trange

def print_hex(all_data):
    result = ''
    for x in all_data:
        result += hex(x)[2:].zfill(2)
    return result

trs = trsfile.open(r'D:\TRAZAS_SERVIO\SBox_Pinata_MaskedSWAES_Rndm_Key.trs')
all_data = np.frombuffer(trs[0].data, dtype=np.uint8)
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
print (all_data[32:50])
print ('Masks:', print_hex(all_data[32:50]))
#print ([hex(x)[2:] for x in all_data[32:50]])
print ('-------------------------------')
print (all_data[50:66])
ciphertext_byte = np.array(all_data[50:66], np.uint8).tobytes()
print ('Cipher text:', print_hex(all_data[50:66]))
expected_ciphertext = AES(bytes(key_byte)).encrypt_block(bytes(plaintext_byte))
print ('cipher text:', binascii.hexlify(bytearray(expected_ciphertext)).decode('ascii'), '(expected)')
print ('-------------------------------')
plaintext_result = AES(bytes(key_byte)).decrypt_block(bytes(ciphertext_byte))
print ('Plain text:', binascii.hexlify(bytearray(plaintext_result)).decode('ascii'))
print ('Plain text:', plaintext, '(expected)')

#print ([hex(x)[2:] for x in all_data[50:66]])


with open('SBox_Pinata_MaskedSWAES_Rndm_Key_wrong_traces.txt', 'w') as file:

    for trace in trange(len(trs)):
        all_data = np.frombuffer(trs[trace].data, dtype=np.uint8)
        numpyct = all_data[50:66]
        plaintext_byte = np.array(all_data[0:16], np.uint8).tobytes()
        expnumpyct = AES(bytes(key_byte)).encrypt_block(bytes(plaintext_byte))
        expnumpyct = np.frombuffer(expnumpyct, dtype=np.uint8)

        for i, element in enumerate(expnumpyct):
            if numpyct[i] != element:
                file.write(str(trace) + '\n')
                break    

trs.close()