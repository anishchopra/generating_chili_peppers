import numpy as np
from PIL import Image
from tqdm import tqdm
import threading

def H(v, x, y):
    s = np.arange(70, dtype=np.float32) + 1
    s = np.repeat(s[None, ...], x.shape[0], axis=0)

    output = {
        'v1': None,
        'v2': None,
    }

    def part1(out):
        u_ = U(s, x, y)
        v_ = V(s, x, y)
        p = P(u_, v_)
        q = Q(p, u_, v_)
        j1 = J(200, q, p, v_)
        j2 = J(2, q, p, v_)
        j20 = J(20, q, p, v_)
        a = A(v, s, x, y, q, p)

        v1 = j1 * a * j2 * (2. / 5. + 3. / 5. * j20)
        out['v1'] = v1

    def part2(out):
        u = s - 1
        exp_term = np.exp(-1000 * (u - 0.5))
        exp1 = np.exp(-exp_term)
        u_ = U(u, x, y)
        v_ = V(u, x, y)
        p = P(u_, v_)
        q = Q(p, u_, v_)
        j200_u = J(200, q, p, v_)
        n_u = N(q, p)

        v2 = (1 - exp1 * j200_u) * (1 - 7./10. * exp1 * n_u)
        v2 = np.repeat(v2[:, None, ...], 70, axis=1)
        iu = np.triu_indices(70, 1)
        v2[:, *iu] = 1.0
        v2 = np.prod(v2, axis=2)
        out['v2'] = v2
    
    t1 = threading.Thread(target=part1, args=(output,))
    t2 = threading.Thread(target=part2, args=(output,))

    threads = [t1, t2]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    v1 = output['v1']
    v2 = output['v2']

    total = np.sum(v1 * v2, axis=1)

    return total

def A(v, s, x, y, q, p):
    b = B(s, q, p)
    cos_term = np.cos((7+v)*s)
    term1 = (5*v - 3*v**2 + 2) * cos_term
    w = W(x, y)
    c20 = C(20, s, q, p, w)
    c5 = C(5, s, q, p, w)
    r = R(s, q, p)
    

    result = 0.1 * (1 - b) * (4*v**2 - 16*v + 16 + np.pow(-1, s) * (11*v - 5*v**2 - 2) + term1 + (1/20.) * (40-s) * (6*c20 + 5*c5 - 2*r)) + \
             0.1 * (-9*v**2 + 13*v + 10 + np.cos((4+v)*s)) * b + (1/20.) * 3 * w
    
    return result

def R(s, q, p):
    exponent = -1 * np.sin((30 + 10 * np.cos(4 * s)) * (q + 0.01 * 3 * np.cos(5 * p)) + 2 * np.cos(17 * q + p + 2) + np.cos(27 * q - 17 * p + 2) + 2 * np.cos(5 * s))
    return np.exp(-1 * np.exp(exponent))

def C(v, s, q, p, w):
    cos_q = np.cos(q)
    abs_cos_q = np.abs(cos_q)
    cos_term = np.abs(np.cos(p - np.sin(0.5 * q + np.pi / 4.) ** 6))
    exponent = -1 * v * cos_term ** (10 + 9 * np.sin(q)) * abs_cos_q ** (4. / 5.) * \
               (0.1 * np.cos((30 + 10 * np.cos(4 * s)) * (q + (3. / 100.) * np.cos(5 * p)) + 2 * np.cos(17 * q + p + 2) + \
                np.cos(27 * q - 17 * p + 2) + 2 * np.cos(5 * s)) + 9. / 10.) + 97 * v / 100 - 3 * v / 100 * w
    return np.exp(-1 * np.exp(exponent))

def B(s, q, p):
    exponent = -100 * (np.cos(0.5 * q + np.pi / 4 + (1 / 20) * np.cos(10 * p + np.cos(3 * s)) ** 4) ** 20 - 0.1)
    return np.exp(-1 * np.exp(exponent))

def N(q, p):
    cos_term = np.abs(np.cos(p - np.sin(0.5 * q + np.pi / 4) ** 6))
    abs_cos_q = np.abs(np.cos(q))
    exponent = -6 * cos_term ** (10 + 9 * np.sin(q)) * abs_cos_q ** (4 / 5) + 96 / 25
    return np.exp(-1 * np.exp(exponent))

def J(v, q, p, v_):
    l = L(v, q, p)
    k = K(v, q, v_)
    return 1 - (1 - l) * (1 - k)

def K(v, q, v_):
    v_cos_term = np.cos(4 * v_ + (7 / 10) * np.cos(0.5 * q + np.pi / 4) ** 30) ** 12
    cos_term = np.cos(0.5 * q + np.pi / 4) ** 30
    exponent1 = -v * (v_cos_term - (97 / 100) - (1 / 50) * cos_term ** 16)
    exponent2 = 40 * (np.abs(cos_term - (23 / 50)) - 7 / 20)
    return np.exp(-1 * np.exp(exponent1) - np.exp(exponent2))

def L(v, q, p):
    cos_term = np.abs(np.cos(p - np.sin(0.5 * q + np.pi / 4) ** 6))
    abs_cos_q = np.abs(np.cos(q))
    exponent = -1 * v * cos_term ** (10 + 9 * np.sin(q)) * abs_cos_q ** (4 / 5) + 7 * v / 10
    return np.exp(-1 * np.exp(exponent))

def Q(p, u, v):
    exponent = 100 * np.cos(p - np.sin(u + np.pi / 4) ** 6)
    return 2 * u + 0.1 * np.cos(3 * v + 2 * u + np.cos(3 * u) + 2) ** 2 + 0.5 * np.exp(-1 * np.exp(exponent))

def P(u, v):
    return 4 * v + 0.1 * np.cos(3 * u + 2 * v + np.cos(4 * v) + 2) ** 2 + 0.1 * np.cos(9 * u - 2 * v + np.cos(6 * v) + 1) ** 2 + \
           0.01 * 3 * np.cos(29 * u + 7 * v + np.cos(5 * v + 8 * u) + 1) ** 4 + 0.01 * np.cos(79 * u - 23 * v + 2 * np.cos(15 * v - 38 * u) + 1) ** 4

def U(s, x, y):
    return np.cos(17 * s) * x + np.sin(17 * s) * y + 2 * np.sin(8 * s)

def V(s, x, y):
    return np.cos(17 * s) * y - np.sin(17 * s) * x + 2 * np.sin(9 * s)

def W(x, y):
    s = np.arange(40, dtype=np.float32) + 1
    s = np.repeat(s[None, ...], x.shape[0], axis=0)
    cos_s = np.cos((11 / 10) ** s * 20 * (np.cos(2 * s) * x + np.sin(2 * s) * y) + 2 * np.sin(5 * s)) ** 2
    cos_s_y = np.cos((11 / 10) ** s * 20 * (np.cos(2 * s) * y - np.sin(2 * s) * x) + 2 * np.sin(6 * s)) ** 2
    exponent = -2 * (cos_s * cos_s_y - 19 / 20)
    v = np.exp(-1 * np.exp(exponent))
    total = np.sum(v, axis=1)[:, None]

    return total

def F(x):
    return np.floor(255 * np.exp(-1 * np.exp(-1000 * x)) * np.abs(x) ** (np.exp(-1 * np.exp(1000 * (x - 1)))))

def rgb(m, n):
    x = (m - 1000) / 600
    y = (601 - n) / 600

    output = [None, None, None]

    def compute(index, x, y, output):
        output[index] = F(H(index, x, y))
    
    threads = [threading.Thread(target=compute, args=(i, x, y, output)) for i in range(3)]
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    
    return np.array(output, np.uint8)

COLS = 2000
ROWS = 1200
# we split the image into chunks so memory usage doesn't explode
NUM_CHUNKS = 1000


if __name__ == '__main__':
    total_pairs = []
    for n in range(ROWS):
        for m in range(COLS):
            total_pairs.append((m, n))
    
    total_pairs = np.array(total_pairs, dtype=np.int32)
    assert len(total_pairs.shape) == 2 and total_pairs.shape[1] == 2
    
    chunk_size = len(total_pairs) // NUM_CHUNKS
    index_chunks = []
    for i in range(NUM_CHUNKS):
        if i < NUM_CHUNKS - 1:
            index_chunks.append(total_pairs[i*chunk_size:i*chunk_size+chunk_size])
        else:
            index_chunks.append(total_pairs[i*chunk_size:])
        
    rgb_chunks = []
    for index_chunk in tqdm(index_chunks):
        rgb_chunks.append(rgb(index_chunk[:, 0, None], index_chunk[:, 1, None]))

    img = np.concatenate(rgb_chunks, axis=1)
    img = img.reshape((3, ROWS, COLS))
    img = img.transpose((1, 2, 0))
    
    img = Image.fromarray(img)
    img.save('chilipeppers.png')