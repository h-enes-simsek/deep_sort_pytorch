import torch

features = torch.load("features.pth")
#gf,gl,qf,ql test.py tarafından üretilen tensorler
qf = features["qf"]
ql = features["ql"]
gf = features["gf"]
gl = features["gl"]



#print(qf.size()) #3368,512
#print(ql.size()) #3368
#print(gf.size()) #19732,512
#print(gl.size()) #19732

#print(qf[45,:].norm()) #kontrol: 45. query resminin feature vektörü normu 1 (normalize edilmiş)
 
#.t() transpose
#.mm() matrix multiplication pytorch method
#bu işlemle query ve gallery setindeki resimlerin featureleri birbiri ile çarpılıyor. (iç çarpım)
#sonuç (query_index,gallery_index) formatında 
scores = qf.mm(gf.t()) #qf x gf.T = 3368,19732
#print(scores[5][10]) #örnek sonuç 5. query resmi ile 10. gallery resminin featurelerinin iç çarpımı

#topk tensordeki en büyük elemanları veriyor, indexleri ile birlikte.
#scores.topk(5, dim=1) satırlardaki en büyük 5 eleman shape=(3368,5) Yani querydeki resimlerle en iyi eşleşen 5 gallery resmi
#[0]bu değerleri ifade ediyor. [1] indisleri seçiyor, yani hangi gallery resmi
#print(scores.topk(5, dim=1)[1].size()) 
res = scores.topk(5, dim=1)[1][:,0] #top1k, [:,0] sadece ilk gallery resimlerinin featurelerinin yerini seçiyor. shape=3368
#print (res.shape)

#gl[res] gallerdeki resim feature'larını top1 gallery indisleriyle seçiyor.
#.eq(ql) seçilmiş galeri featureleriyle query featurelerını kıyaslıyor. true false olarak yeniden yazıyor.
#.sum.item ile true değerler toplanıyor.
top1correct = gl[res].eq(ql).sum().item()

#doğru query-galeri resim eşleşmesi / query boyutu
print("Acc top1:{:.3f}".format(top1correct/ql.size(0))) 
#ZQ Pei'nin weigthleriyle 0.985, Market1501 ile 40 epoch eğitilmiş weightlerle 0.986



#makaleye koymak için örnek 3x3 sonuçlar
#gallerydeki ilk 6617 resim junk (-1 ve 0 class), sonrakiler labeller query ile karşılaştırmak için uygun
#class 1 3 4, query de sırası ile 6 5 ve ? resim var, galleride 59 10 ve ? sayıda resim var.
print(scores[0][6617+5]) #query class 1 içinde 1. resim ve gallery class 1 içinde 6. resim
print(scores[0][6617+59+5]) #query class 1 içinde 1. resim ve gallery class 3 içinde 6. resim
print(scores[0][6617+59+10+5]) #query class 1 içinde 1. resim ve gallery class 4 içinde 6. resim

print(scores[0+6][6617+5])#query class 3 içinde 1. resim ve gallery class 1 içinde 6. resim
print(scores[0+6][6617+59+5])  #query class 3 içinde 1. resim ve gallery class 3 içinde 6. resim
print(scores[0+6][6617+59+10+5]) #query class 3 içinde 1. resim ve gallery class 4 içinde 6. resim

print(scores[0+6+5][6617+5]) #query class 4 içinde 1. resim ve gallery class 1 içinde 6. resim
print(scores[0+6+5][6617+59+5]) #query class 4 içinde 1. resim ve gallery class 3 içinde 6. resim
print(scores[0+6+5][6617+59+10+5]) #query class 4 içinde 1. resim ve gallery class 4 içinde 6. resim


