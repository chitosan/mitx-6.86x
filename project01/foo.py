import numpy as np
import random

def hinge_l(z):
    if z >= 1.0:
        return 0.0
    if z < 1.0:
        return 1 - z
    raise NotImplementedError
def hinge_loss_single(feature_vector, label, theta, theta_0):
    z = label * (np.dot(theta, feature_vector) + theta_0)
    if z >= 1.0:
        return 0.0
    if z < 1.0:
        return 1 - z
    raise NotImplementedError

def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    def hinge(x):
        return max(0., 1. - x)
    n = float(np.size(labels))
    z = labels * (np.dot(theta, feature_matrix.T) + theta_0)
    loss = np.vectorize(hinge)(z)
    average = np.sum(loss)/n
    return average

def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    theta, theta_0 = current_theta, current_theta_0
    z = label * np.dot(theta, feature_vector) + theta_0
    if z <= 1e-8:
        theta = theta + label * feature_vector
        theta_0 = theta_0 + label
    return theta, theta_0
    raise NotImplementedError

def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices

def perceptron(feature_matrix, labels, T):
    size = feature_matrix.shape[1]
    theta = np.zeros((size,))
    theta_0 = 0.0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            (theta, theta_0) = perceptron_single_step_update(
                feature_matrix[i, :],
                labels[i],
                theta,
                theta_0)
            pass
    return (theta, theta_0)
    raise NotImplementedError

def average_perceptron(feature_matrix, labels, T):
    size = feature_matrix.shape[1]
    sum_theta = np.zeros((size,))
    theta = np.zeros((size,))
    sum_theta_0 = 0.0
    theta_0 = 0.0
    for t in range(T):
        n = 0
        for i in get_order(feature_matrix.shape[0]):
            (theta, theta_0) = perceptron_single_step_update(
                feature_matrix[i, :],
                labels[i],
                theta,
                theta_0)
            n = n + 1
            sum_theta = sum_theta + theta
            sum_theta_0 = sum_theta_0 + theta_0
    average_theta = sum_theta/(T*n)
    average_theta_0 = sum_theta_0/(T*n)
    return (average_theta, average_theta_0)
    raise NotImplementedError

def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        theta,
        theta_0):
    if label*(np.dot(theta,feature_vector)) <= 1.0:
        theta = (1-eta*L)*theta + (eta*label)*feature_vector
    else:
        theta = (1-eta*L)*theta
    return theta, theta_0
    raise NotImplementedError



x0 = np.array([ 0, 2, 3, 0, 2, 5, 5, 2, 4, 5])
x1 = np.array([ 0, 0, 0, 2, 2, 1, 2, 4, 4, 5])
y  = np.array([-1,-1,-1,-1,-1, 1, 1, 1, 1, 1])
f = (np.stack((x0,x1),axis=-1))

#for i in range(50):
#    print(i,perceptron(f,y,i))

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import hinge_loss

def tranformation2d3d(x):
    l = len(x)
    psi = np.zeros([l,3])
    for i in range(l):
        a = x[i][0] ** 2
        b = (2 ** 0.5) * (x[i][0]) * (x[i][1])
        c = x[i][1] ** 2
        psi[i]=[a,b,c]
    return psi

# Input data
X = np.array([[0,0], [2,0], [3,0], [0,2], [2,2], [5,1], [5,2], [2,4], [4,4], [5,5]])
y = np.array([-1,-1,-1,-1,-1, 1, 1, 1, 1, 1])

# Train SVM with linear kernel
clf = SVC(kernel='linear', C=10)
clf.fit(X, y)
# Get parameters and corresponding
w1, w2 = clf.coef_[0]
b = clf.intercept_

print("Parameters = ",w1, w2, b)
print("Corresponding = ", -b/w1, -b/w2)
print((np.dot([w1,w2],X.T)+b)/np.linalg.norm([w1,w2]))
print("hinge LOSS")
z = (np.dot([w1,w2],X.T)+b)
print(z)
l = np.vectorize(hinge_l)(z)
print(l)
print(np.sum(l))
print("---------------------------------")
y =np.array([-1,-1,-1,-1,-1,+1,+1,+1,+1,+1])
X= np.array([[0,0],[2,0],[1,1],[0,2],[3,3],[4,1],[5,2],[1,4],[4,4],[5,5]])
psi = tranformation2d3d(X)
print("X")
print(X)
print("psi")
print(psi)
clf = SVC(kernel='linear', C=100)
clf.fit(psi, y)
# Get parameters and corresponding
w1, w2, w3 = clf.coef_[0]
b = clf.intercept_
print("Parameters = ",w1, w2, w3, b)
print("Corresponding = ", -b/w1, -b/w2)
print((np.dot([w1,w2],X.T)+b)/np.linalg.norm([w1,w2]))
print("hinge LOSS")

print("---------------------------------")

clf = SVC(kernel="poly", degree=2)
clf.fit(X, y)

print("Parameters = ",clf.coef_,w1, w2, b)
print("Corresponding = ", -b/w1, -b/w2)
print((np.dot([w1,w2],X.T)+b)/np.linalg.norm([w1,w2]))
print("hinge LOSS")
z = (np.dot([w1,w2],X.T)+b)
print(z)
l = np.vectorize(hinge_l)(z)
print(l)
print(np.sum(l))


'''
HINGE LOSS DATA

Problem Variables...
Feature Vector: [0.01977061 0.75717366 0.32463439 0.68371134 0.37835374 0.15041504
 0.65427149 0.62436318 0.83046868 0.83671147]
Label: 1.0
Theta: [2.52900637 0.06603505 0.15401942 0.07313028 0.13215146 0.33241356
 0.07642088 0.0800816  0.06020697 0.05975776]
Theta_0: 0.5
singleHingeLoss Value is  0.0000000
Test completed

PERCEPTRON DATA

Output:
perceptron_single_step_update input:
feature_vector: [ 0.10470908  0.20299147  0.30580197  0.16200104  0.00277622  0.08109208
 -0.30994275 -0.49095164  0.12976674  0.49881033]
label: 1
theta: [-0.48511876 -0.24085429  0.48397298  0.20914529  0.15989067 -0.37766278
  0.07806714 -0.33158527  0.25010663 -0.40735737]
theta_0: -0.25219390537076125
perceptron_single_step_update output is (['-0.3804097', '-0.0378628', '0.7897749', '0.3711463', '0.1626669', '-0.2965707', '-0.2318756', '-0.8225369', '0.3798734', '0.0914530'], '0.7478061')
Test: perceptron single step update 2
Test for correct prediction

Your output:
perceptron_single_step_update input:
feature_vector: [ 0.00172751 -0.39030874  0.22159702 -0.39442814 -0.13694739 -0.13625332
  0.096334   -0.28624113  0.23964461 -0.36465328]
label: -1
theta: [ 0.31019718 -0.41685395  0.43400157 -0.04242831 -0.09491979 -0.29568427
  0.25498497 -0.2990711  -0.12983148  0.1519356 ]
theta_0: -0.9443970834547866
perceptron_single_step_update output is (['0.3084697', '-0.0265452', '0.2124045', '0.3519998', '0.0420276', '-0.1594309', '0.1586510', '-0.0128300', '-0.3694761', '0.5165889'], '-1.9443971')
-----
Correct output:
perceptron_single_step_update input:
feature_vector: [ 0.00172751 -0.39030874  0.22159702 -0.39442814 -0.13694739 -0.13625332
  0.096334   -0.28624113  0.23964461 -0.36465328]
label: -1
theta: [ 0.31019718 -0.41685395  0.43400157 -0.04242831 -0.09491979 -0.29568427
  0.25498497 -0.2990711  -0.12983148  0.1519356 ]
theta_0: -0.9443970834547866
perceptron_single_step_update output is (['0.3101972', '-0.4168539', '0.4340016', '-0.0424283', '-0.0949198', '-0.2956843', '0.2549850', '-0.2990711', '-0.1298315', '0.1519356'], '-0.9443971')
Test: perceptron single step update 3
Test for incorrect prediction

Output:
perceptron_single_step_update input:
feature_vector: [-0.35804173  0.45105554  0.12115707  0.13282913 -0.34093641 -0.10748029
  0.21723838  0.07919139 -0.0583864   0.18331289]
label: 1
theta: [ 0.48061322 -0.13025438 -0.24532365  0.19592721  0.31875087  0.46142489
 -0.14278747  0.03011105 -0.0185533   0.16991932]
theta_0: -0.15352906605047767
perceptron_single_step_update output is (['0.1225715', '0.3208012', '-0.1241666', '0.3287563', '-0.0221855', '0.3539446', '0.0744509', '0.1093024', '-0.0769397', '0.3532322'], '0.8464709')
Test: perceptron single step update 4
Test for boundary case for positive label

Output:
perceptron_single_step_update input:
feature_vector: [ 0.46027926  0.29775937  0.11246961 -0.12378052  0.41166378  0.07630349
  0.31298193  0.00158826 -0.04238583  0.4404474 ]
label: 1
theta: [ 0.10711067 -0.40319219 -0.36228182 -0.01972854  0.16491187 -0.04609795
 -0.47785078  0.1073763  -0.13903493  0.20765111]
theta_0: 0.09672192477973104
perceptron_single_step_update output is (['0.5673899', '-0.1054328', '-0.2498122', '-0.1435091', '0.5765756', '0.0302055', '-0.1648689', '0.1089646', '-0.1814208', '0.6480985'], '1.0967219')
Test: perceptron single step update 5
Test for boundary case for negative label

Output:
perceptron_single_step_update input:
feature_vector: [ 0.22222037  0.02890427 -0.05068555 -0.31366039  0.16627174 -0.46972173
  0.44369192 -0.06428105  0.31402111 -0.45705511]
label: -1
theta: [ 0.42638128  0.33594382 -0.01263249 -0.27254585 -0.13824211 -0.38893684
 -0.03403267  0.13933304 -0.04543297 -0.14020006]
theta_0: -0.37605001524039555
perceptron_single_step_update output is (['0.2041609', '0.3070395', '0.0380531', '0.0411145', '-0.3045138', '0.0807849', '-0.4777246', '0.2036141', '-0.3594541', '0.3168550'], '-1.3760500')

---

AVERAGE PERCEPTRON

Test: average perceptron 1
Test high dimension

Output:
average_perceptron input:
feature_matrix: [[-0.25288861  0.40003046 -0.246615    0.29049142  0.10828785  0.46889963
   0.43365275  0.03607241  0.34729951  0.18652895]
 [ 0.37887098 -0.33996972 -0.11409327  0.42151187  0.04936691  0.1059058
  -0.41364331  0.47773271 -0.02483739  0.12543317]
 [-0.34381769 -0.48308484 -0.11040539  0.21714299 -0.02102351  0.30030446
   0.03534012 -0.10863231  0.46169879  0.19345152]
 [-0.31857633 -0.28569085  0.29606906 -0.49534256  0.19361805 -0.31942071
   0.39772036  0.34104673 -0.08174699 -0.47822401]
 [ 0.17573615 -0.45610358  0.26990172  0.12064697 -0.27436722  0.33193679
  -0.17761403 -0.41934737 -0.11935083  0.30194325]]
labels: [-1  1  1  1  1]
T: 5
average_perceptron output is ['0.1460446', '-1.0092542', '0.7020708', '-0.4998262', '-0.2181793', '-0.3046986', '-0.2396149', '-0.1904527', '-0.4732101', '-0.2181913']
Test: average perceptron 2
Test high data number

Output:
average_perceptron input:
feature_matrix: [[ 0.40575689 -0.41311546  0.16990972 -0.40820196 -0.32460829]
 [ 0.22638442 -0.08163622  0.08353305 -0.10355208  0.38437806]
 [-0.17777628 -0.10555374 -0.43048503 -0.00097867 -0.01568204]
 [-0.43938539 -0.13275607 -0.48885518 -0.4420344   0.1669595 ]
 [-0.17659976 -0.05610784  0.0245264   0.41858329  0.4048259 ]
 [-0.41791922 -0.09369963  0.20444056  0.4955068   0.4697723 ]
 [-0.2898955  -0.24570343  0.44407277 -0.00539271  0.25165835]
 [-0.47692161 -0.35629081  0.06109791  0.36866488  0.26970802]
 [ 0.26987977  0.42612681 -0.17961814 -0.24816257  0.17376996]
 [-0.38097717  0.23865764  0.1751625  -0.33580229 -0.47577881]]
labels: [-1  1  1 -1 -1  1 -1  1  1 -1]
T: 5
average_perceptron output is ['0.2450990', '0.2548699', '-1.6289264', '0.7278253', '0.7165107']
Test: average perceptron 3
Test whether update exactly once when T = 1

Output:
average_perceptron input:
feature_matrix: [[ 0.29086972 -0.4792536   0.19340556  0.30653372  0.31120714 -0.08992844
  -0.33070521  0.45821293  0.36086579 -0.43920485]
 [-0.27990114 -0.33829364  0.32713177 -0.25617657  0.28080931  0.25565689
  -0.13729724  0.37499199 -0.17022876  0.38118265]
 [-0.10402415 -0.43059823  0.02759074 -0.15096348  0.04180791  0.23356457
  -0.1637117   0.40656989 -0.20819206 -0.47375404]
 [ 0.34849682 -0.28454771  0.01241686 -0.01782604  0.35803954 -0.30828623
   0.19207243  0.14820933  0.42890694  0.15475506]
 [-0.15728864  0.1922825  -0.31312896 -0.05110992 -0.49397945  0.17923943
  -0.40562555  0.16996071 -0.17853926  0.40028579]
 [-0.41816017  0.22286189  0.48715658 -0.20629836 -0.00270493  0.05418993
  -0.07977075 -0.11700681 -0.25603272  0.34712654]
 [-0.41816017  0.22286189  0.48715658 -0.20629836 -0.00270493  0.05418993
  -0.07977075 -0.11700681 -0.25603272  0.34712654]
 [ 0.40989579 -0.23200905  0.49922641 -0.47215887  0.05941742 -0.01070731
   0.03840202  0.23687708 -0.17809311 -0.15768021]
 [ 0.30420506  0.20911881 -0.48370273  0.45005448 -0.36571904 -0.2882422
   0.16132368 -0.19460665 -0.16603391  0.11897396]
 [-0.26459914  0.25087172 -0.41871637 -0.05499894  0.01628657 -0.44582761
   0.23601507 -0.26225949 -0.37246537  0.0250584 ]]
labels: [-1  1  1  1 -1 -1  1 -1 -1  1]
T: 1
average_perceptron output is ['-0.8290118', '0.0754850', '-0.3401460', '-0.1860235', '0.4222668', '-0.2166771', '0.1129111', '-0.1528824', '-0.0331451', '0.0425350']
Test: average perceptron 4
Test when it can converge

Output:
average_perceptron input:
feature_matrix: [[ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
   0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
   0.00000000e+00  0.00000000e+00]
 [ 2.32765854e-01  6.65890310e-01  5.64539140e-01  3.96343096e-01
   3.48476815e-01  7.35314489e-01  2.17066703e-01  6.07529740e-02
   8.08920640e-01  2.93608304e-01]
 [-3.42651746e-01 -7.26964710e-02 -8.55662908e-01 -4.17049187e-01
  -8.60646198e-01 -3.13058858e-02 -9.56044772e-01 -4.06194191e-01
  -4.92181940e-01 -7.45482265e-01]
 [ 8.71266624e-01  5.66933889e-01  7.42427832e-01  7.96564932e-01
   9.91177648e-01  6.37682315e-01  5.84276835e-01  6.93057433e-01
   8.73446672e-01  7.92621051e-01]
 [-9.65052297e-01 -3.35487768e-01 -7.04345649e-01 -3.15442191e-01
  -1.63171118e-01 -9.13523715e-02 -1.73359161e-01 -5.70107258e-01
  -3.55863691e-01 -1.69959722e-01]
 [ 5.88464021e-02  5.18464763e-01  6.11018551e-01  2.55127491e-01
   5.20938633e-01  4.84265432e-01  3.83154516e-04  8.97327286e-01
   5.03083605e-01  7.74249735e-01]
 [-7.20313442e-01 -6.89482901e-01 -5.73718709e-01 -8.04365354e-01
  -3.14958506e-01 -4.53730738e-01 -7.30690590e-01 -5.95506817e-01
  -3.94437805e-01 -9.05626397e-01]
 [ 5.81400253e-01  2.69643734e-01  9.55744674e-01  6.48900565e-01
   5.58997902e-01  1.98216127e-01  9.82531149e-01  4.78561758e-01
   4.97312897e-01  9.11911587e-01]
 [ 2.70623812e-01  6.26315453e-01  4.98807964e-01  4.34598850e-01
   3.99953347e-01  1.80165175e-01  4.01831486e-01  8.70176914e-01
   3.90920658e-01  6.92308201e-01]
 [-6.13829514e-01 -1.54606267e-01 -8.95701085e-01 -9.53519966e-01
  -6.96661603e-01 -2.12227607e-01 -8.73416371e-01 -3.65263130e-01
  -1.60740972e-01 -2.95105520e-01]]
labels: [-1  1 -1  1 -1  1 -1  1  1 -1]
T: 100
average_perceptron output is ['0.6138295', '0.1546063', '0.8957011', '0.9535200', '0.6966616', '0.2122276', '0.8734164', '0.3652631', '0.1607410', '0.2951055']
Test: average perceptron 5
Test when it cannot converge

Output:
average_perceptron input:
feature_matrix: [[-0.21610709 -0.29414619  0.23147192 -0.14710005 -0.37367388 -0.3532686
   0.09324123  0.19048527  0.25356815  0.03249604]
 [-0.44917473  0.464259    0.00469458  0.17845459  0.45547988  0.49408279
   0.22272871  0.08565401 -0.42948581  0.05376368]
 [-0.20593517  0.16860696 -0.40975033 -0.04979148  0.26297689  0.40726457
  -0.1708299   0.24178269  0.32349658 -0.29461259]
 [-0.14190223 -0.34631013  0.20217694 -0.08939114  0.01563256 -0.24361794
  -0.26207129  0.00853995  0.44500064  0.25284787]
 [ 0.35274419  0.15270493  0.23444042 -0.3881418  -0.4643564  -0.20295211
   0.04689588  0.26661437 -0.06380267  0.17903774]
 [-0.47322061  0.27053324 -0.1766483  -0.26746725  0.25595011  0.23565614
  -0.35519383  0.33810369 -0.45407728  0.40695708]
 [ 0.06938845  0.3370708  -0.27171327  0.38892342 -0.26312978  0.20957848
  -0.29934358 -0.39727662 -0.10470943 -0.48033701]
 [-0.14190223 -0.34631013  0.20217694 -0.08939114  0.01563256 -0.24361794
  -0.26207129  0.00853995  0.44500064  0.25284787]
 [-0.41811893 -0.30765593 -0.35596231  0.19379607  0.41238472 -0.21084179
  -0.2024383   0.0350493   0.49391038  0.01586308]
 [ 0.33506576 -0.20885652 -0.46003352  0.27165654  0.4209158   0.36504807
   0.32019415  0.40211099  0.38053274 -0.47113879]]
labels: [-1 -1  1  1 -1  1 -1 -1 -1  1]
T: 100
average_perceptron output is ['1.2092683', '-1.5347041', '-0.2161610', '-1.6647843', '1.4447101', '2.0093419', '-2.0269418', '2.0166226', '-0.1259048', '1.5355935']

-------
CODE

print("hinge single")
feature_vector = np.array([1, 2])
label, theta, theta_0 = 1, np.array([-1, 1]), -0.2
exp_res = 1 - 0.8
print("single")
print(hinge_loss_single(feature_vector, label, theta, theta_0))

print("hinge full")
feature_vector = np.array([[1, 2], [1, 2]])
label, theta, theta_0 = np.array([1, 1]), np.array([-1, 1]), -0.2
exp_res = 1 - 0.8
print(hinge_loss_full(feature_vector, label, theta, theta_0))

print("Perceptron single update")
feature_vector = np.array([1, 2])
label, theta, theta_0 = 1, np.array([-1, 1]), -1.5
exp_res = (np.array([0, 3]), -0.5)
print(perceptron_single_step_update(feature_vector, label, theta, theta_0))
print("")
print("Perceptron single step update")
print("   test for correct prediction")
feature_vector = np.array([0.30549118,0.19996728,0.04141735,0.49155461,0.2594334,-0.09620987, 0.36465185,-0.04578536,-0.45284302,0.19927198])
label = -1
theta = np.array([-0.44797526,-0.04747628,-0.21997313, 0.21905479,-0.36123058,-0.20570137,-0.29134304,-0.21047259,-0.30421932,-0.13952193])
theta_0 = -0.24305211435764226
exp_result, exp_theta_0 = (np.array([-0.4479753, -0.0474763, -0.2199731, 0.2190548, -0.3612306, -0.2057014,-0.2913430, -0.2104726, -0.3042193,-0.1395219]), -0.2430521)
result, theta_0 = perceptron_single_step_update(feature_vector, label, theta, theta_0)
print(result, theta_0)
print(exp_result, exp_theta_0)
print(np.allclose(result, exp_result))
print("****************************************************************")
print("--- test for random feature and theta")
feature_vector = np.array([ 0.10470908, 0.20299147,0.30580197,0.16200104,0.00277622,0.08109208,-0.30994275,-0.49095164,0.12976674,0.49881033])
label = 1
theta = np.array([-0.48511876,-0.24085429,0.48397298,0.20914529,0.15989067,-0.37766278,0.07806714,-0.33158527,0.25010663,-0.40735737])
theta_0 = -0.25219390537076125
exp_result, exp_theta_0 = (np.array([-0.3804097,-0.0378628, 0.7897749, 0.3711463, 0.1626669,-0.2965707,-0.2318756,-0.8225369, 0.3798734, 0.0914530]), 0.7478061)
result, theta_0 = perceptron_single_step_update(feature_vector, label, theta, theta_0)
print(result, theta_0)
print(exp_result, exp_theta_0)
print(np.allclose(result, exp_result))
print("--- test for correct prediction")
feature_vector = np.array([0.00172751,-0.39030874, 0.22159702,-0.39442814,-0.13694739,-0.13625332, 0.096334,-0.28624113, 0.23964461,-0.36465328])
label = -1
theta = np.array([0.31019718,-0.41685395, 0.43400157,-0.04242831,-0.09491979,-0.29568427, 0.25498497,-0.2990711,-0.12983148, 0.1519356])
theta_0 = -0.9443970834547866
exp_result, exp_theta_0 = (np.array([0.3101972,-0.4168539, 0.4340016,-0.0424283,-0.0949198,-0.2956843,0.2549850,-0.2990711,-0.1298315,0.1519356]),-0.9443971)
result, theta_0 = perceptron_single_step_update(feature_vector, label, theta, theta_0)
print(result, theta_0)
print(exp_result, exp_theta_0)
print(np.allclose(result, exp_result))
print("--- test for incorrect prediction")
feature_vector = np.array([-0.35804173, 0.45105554, 0.12115707, 0.13282913,-0.34093641,-0.10748029, 0.21723838, 0.07919139,-0.0583864, 0.18331289])
label = 1
theta = np.array([ 0.48061322,-0.13025438,-0.24532365, 0.19592721, 0.31875087, 0.46142489,-0.14278747, 0.03011105,-0.0185533, 0.16991932])
theta_0 = -0.15352906605047767
exp_result, exp_theta_0 = (np.array([0.1225715, 0.3208012,-0.1241666, 0.3287563,-0.0221855, 0.3539446, 0.0744509, 0.1093024,-0.0769397, 0.3532322]),0.8464709)
result, theta_0 = perceptron_single_step_update(feature_vector, label, theta, theta_0)
print(result, theta_0)
print(exp_result, exp_theta_0)
print(np.allclose(result, exp_result))
print("--- Test for boundary case for positive label")
feature_vector = np.array([0.46027926, 0.29775937, 0.11246961,-0.12378052, 0.41166378, 0.07630349, 0.31298193, 0.00158826,-0.04238583, 0.4404474])
label = 1
theta = np.array([0.10711067,-0.40319219,-0.36228182,-0.01972854, 0.16491187,-0.04609795,-0.47785078, 0.1073763,-0.13903493, 0.20765111])
theta_0 = 0.09672192477973104
exp_result, exp_theta_0 = (np.array([0.5673899,-0.1054328,-0.2498122,-0.1435091,0.5765756,0.0302055,-0.1648689,0.1089646,-0.1814208,0.6480985]), 1.0967219)
result, theta_0 = perceptron_single_step_update(feature_vector, label, theta, theta_0)
print(result, theta_0)
print(exp_result, exp_theta_0)
print(np.allclose(result, exp_result))
print("--- Test for boundary case for negative label")
feature_vector = np.array([ 0.22222037, 0.02890427,-0.05068555,-0.31366039, 0.16627174,-0.46972173, 0.44369192,-0.06428105, 0.31402111,-0.45705511])
label = -1
theta = np.array([0.42638128, 0.33594382,-0.01263249,-0.27254585,-0.13824211,-0.38893684,-0.03403267, 0.13933304,-0.04543297,-0.14020006])
theta_0 = -0.37605001524039555
exp_result, exp_theta_0 = (np.array([0.2041609, 0.3070395, 0.0380531, 0.0411145,-0.3045138, 0.0807849,-0.4777246, 0.2036141,-0.3594541, 0.3168550]), -1.3760500)
result, theta_0 = perceptron_single_step_update(feature_vector, label, theta, theta_0)
print(result, theta_0)
print(exp_result, exp_theta_0)
print(np.allclose(result, exp_result))
print("")

print("Full perceptron")
print("--- ")
feature_vector = np.array([[0.19018413,-0.27797033,-0.12522691,0.18233594,-0.10275992,-0.13346594,-0.44511572,0.14549705,0.04007474,0.00136917],
                           [-0.38737393,0.36835259,0.43914363,-0.25077652,0.36479819,-0.2201786,-0.05775183,0.36847621,-0.20243208,-0.11895794],
                           [-0.34229924,-0.46851685,0.26869282,0.179921,0.23157122,0.48307805,0.35272901,-0.47541196,-0.14875796,0.26869951],
                           [-0.34628031,0.26523403,0.00158151,-0.09882203,0.49131234,0.44508969,-0.0754451,-0.24823076,0.1624609,-0.28752572],
                           [0.05879913,0.01155871,-0.38006761,0.28095214,-0.20293743,0.46175998,0.00133664,-0.46263215,-0.21510828,0.18391205]])
exp_result = np.array([0.2246387,1.606474,-0.7331503,-0.2310502,-0.4014592,-0.1293488,-0.3154209,0.2485388,-0.3752075,-0.289902])
label = np.array([-1, 1, 1,-1, 1])
result, theta_0 = perceptron(feature_vector, label, 10)
print(result, theta_0)
print(exp_result, theta_0)
print(np.allclose(result, exp_result))

print(average_perceptron(feature_vector,label,5))

'''