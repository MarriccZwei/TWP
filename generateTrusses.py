from Classes import hitboxes as hbx
from Classes import strutEles as se

'''Import Joint Coordinates'''
trImp = "3494.998;1.554;197.366|3467.09;929.563;289.157|3438.623;1876.134;382.784|3410.157;2822.704;476.41|3381.69;3769.275;570.037|3353.224;4715.845;663.664|3324.758;5662.415;757.29|3296.291;6608.986;850.917|3267.825;7555.556;944.544|3239.358;8502.127;1038.17|3210.892;9448.697;1131.797|3182.426;10395.267;1225.424|3153.959;11341.838;1319.05|3125.493;12288.408;1412.677|3097.026;13234.979;1506.304|3068.56;14181.549;1599.93|3040.094;15128.119;1693.557|3011.627;16074.69;1787.184|2983.161;17021.26;1880.81|2954.695;17967.831;1974.437"
tfImp = "750;1.554;339.263|787.469;835.553;418.068|830.11;1784.673;507.746|872.751;2733.793;597.42|915.392;3682.914;687.089|958.033;4632.036;776.752|1000.674;5581.158;866.408|1043.315;6530.281;956.058|1085.956;7479.405;1045.699|1128.597;8428.53;1135.332|1171.238;9377.656;1224.955|1213.88;10326.783;1314.566|1256.521;11275.911;1404.164|1299.162;12225.041;1493.748|1341.804;13174.173;1583.314|1384.445;14123.307;1672.861|1427.087;15072.443;1762.385|1469.729;16021.582;1851.882|1512.371;16970.724;1941.348|1555.013;17919.87;2030.775"
bfImp = "750;-1.554;-155.371|789.836;885.142;-57.397|832.407;1832.696;47.303|874.978;2780.249;152.005|917.548;3727.802;256.707|960.119;4675.355;361.41|1002.689;5622.908;466.115|1045.26;6570.461;570.822|1087.83;7518.014;675.53|1130.401;8465.566;780.239|1172.971;9413.118;884.951|1215.542;10360.67;989.665|1258.112;11308.222;1094.381|1300.683;12255.774;1199.099|1343.253;13203.325;1303.821|1385.824;14150.876;1408.546|1428.394;15098.426;1513.274|1470.964;16045.976;1618.006|1513.535;16993.526;1722.742|1556.105;17941.075;1827.482"
brImp = "3494.998;-1.554;-130.128|3466.014;962.219;-24.709|3437.578;1907.77;78.717|3409.143;2853.321;182.143|3380.707;3798.871;285.569|3352.271;4744.422;388.995|3323.835;5689.973;492.421|3295.4;6635.524;595.848|3266.964;7581.074;699.274|3238.528;8526.625;802.7|3210.092;9472.176;906.126|3181.657;10417.727;1009.552|3153.221;11363.277;1112.978|3124.785;12308.828;1216.404|3096.35;13254.379;1319.831|3067.914;14199.929;1423.257|3039.478;15145.48;1526.683|3011.042;16091.031;1630.109|2982.607;17036.582;1733.535|2954.171;17982.132;1836.961"

trList = hbx.uncodePoints(trImp)
tfList = hbx.uncodePoints(tfImp)
bfList = hbx.uncodePoints(bfImp)
brList = hbx.uncodePoints(brImp)

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
rout, rin = 15, 10 #for now

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
    trusses[f"dd{m}"]=se.Truss(f"bf{m-1}", f"tr{m}", rout, rin)

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
