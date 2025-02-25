from Classes import hitboxes as hbx
from Classes import strutEles as se

'''Import Joint Coordinates'''
ftsFbsRtsRbs = "999.098;0;458.239|1047.717;1328.625;578.342|1112.3;3093.524;737.864|1176.884;4858.426;897.361|1246.315;6755.786;1068.795|1314.587;8621.456;1237.325|1383.438;10502.979;1407.235|1452.29;12384.508;1577.08|1521.142;14266.045;1746.841|1589.994;16147.594;1916.49|1661.149;18092.06;2091.649&998.893;0;-153.848|1049.643;1386.864;-0.973|1114.099;3148.279;193.195|1178.555;4909.693;387.372|1247.849;6803.3;596.134|1315.985;8665.273;801.423|1384.7;10543.06;1008.474|1453.414;12420.845;1215.549|1522.129;14298.626;1422.657|1590.843;16176.403;1629.811|1661.855;18116.947;1843.957&3494.247;0;336.507|3454.177;1332.414;462.624|3401.112;3096.943;629.641|3348.047;4861.472;796.658|3291;6758.428;976.21|3234.905;8623.697;1152.762|3178.334;10504.809;1330.814|3121.763;12385.922;1508.866|3065.192;14267.034;1686.918|3008.621;16148.147;1864.97|2950.159;18092.145;2048.974&3493.711;0;-32.671|3452.585;1367.521;113.506|3399.585;3129.9;301.891|3346.584;4892.28;490.276|3289.606;6786.925;692.799|3233.58;8649.922;891.939|3177.078;10528.743;1092.77|3120.576;12407.564;1293.602|3064.074;14286.385;1494.433|3007.571;16165.206;1695.265|2949.18;18106.836;1902.81"
edgesSeparated = ftsFbsRtsRbs.split('&')

tfList = hbx.uncodePoints(edgesSeparated[0])
bfList = hbx.uncodePoints(edgesSeparated[1])
trList = hbx.uncodePoints(edgesSeparated[2])
brList = hbx.uncodePoints(edgesSeparated[3])


ribCount = len(trList) #cuz all 4 lens shall be same

'''Generating Joints'''
jointDict = dict()
for i in range(len(trList)):
    jointDict[f"tr{i}"] = se.Joint(trList[i])
for j in range(len(tfList)):
    jointDict[f"tf{j}"] = se.Joint(tfList[j])
for k in range(len(brList)):
    jointDict[f"br{k}"] = se.Joint(brList[k])
for l in range(len(bfList)):
    jointDict[f"bf{l}"] = se.Joint(bfList[l])

'''Generating Trusses'''
trusses = dict()
rout, rin = 25, 20 #for now

#first the root trusses
trusses["t0"]=se.Truss("tr0", "tf0", rout, rin)
trusses["f0"]=se.Truss("tf0", "bf0", rout, rin)
trusses["b0"]=se.Truss("br0", "bf0", rout, rin)
trusses["r0"]=se.Truss("tr0", "br0", rout, rin)
trusses["d0"]=se.Truss("br0", "tf0", rout, rin)

for m in range(1, ribCount):
    #rib trusses at every other rib
    trusses[f"t{m}"]=se.Truss(f"tr{m}", f"tf{m}", rout, rin)
    trusses[f"f{m}"]=se.Truss(f"tf{m}", f"bf{m}", rout, rin)
    trusses[f"b{m}"]=se.Truss(f"br{m}", f"bf{m}", rout, rin)
    trusses[f"r{m}"]=se.Truss(f"tr{m}", f"br{m}", rout, rin)
    trusses[f"d{m}"]=se.Truss(f"br{m}", f"tf{m}", rout, rin)

    #longitudinal trusses
    trusses[f"tr{m}"]=se.Truss(f"tr{m-1}", f"tr{m}", rout, rin)
    trusses[f"tf{m}"]=se.Truss(f"tf{m-1}", f"tf{m}", rout, rin)
    trusses[f"br{m}"]=se.Truss(f"br{m-1}", f"br{m}", rout, rin)
    trusses[f"bf{m}"]=se.Truss(f"bf{m-1}", f"bf{m}", rout, rin)
    trusses[f"dt{m}"]=se.Truss(f"tf{m-1}", f"tr{m}", rout, rin)
    trusses[f"dr{m}"]=se.Truss(f"tr{m-1}", f"br{m}", rout, rin)
    trusses[f"df{m}"]=se.Truss(f"tf{m-1}", f"bf{m}", rout, rin)
    trusses[f"db{m}"]=se.Truss(f"bf{m-1}", f"br{m}", rout, rin)
    #trusses[f"dd{m}"]=se.Truss(f"bf{m-1}", f"tr{m}", rout, rin)

#connecting trusses to the node coords
for truss in trusses.values():
    truss.init_len(jointDict)

#exporting trusses to vs format
trussString = ""
for truss in trusses.values():
    j1 = jointDict[truss.node1key]
    j2 = jointDict[truss.node2key]
    trussString += f"{j1.x};{j1.y};{j1.z};{j2.x};{j2.y};{j2.z};{rout};{rin}|"
trussString = trussString[:-1]
print(trussString)
