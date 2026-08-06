from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image, ImageDraw, ImageFont

OUT = Path(r"F:\Challenger program\多传感器环境感知项目工作情况介绍.docx")
ASSETS = Path(r"F:\Challenger program\project_overview_assets")
ASSETS.mkdir(exist_ok=True)

NAVY = "17365D"; BLUE = "2E74B5"; TEAL = "0F766E"; LIGHT = "EAF2F8"; GRAY = "F2F4F7"; INK = "1F2937"; MUTED = "5B6470"

def font(size, bold=False, color=INK, name="Microsoft YaHei"):
    return (name, size, bold, color)

def get_font(sz, bold=False):
    for name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        try: return ImageFont.truetype(name, sz, index=0)
        except Exception: pass
    return ImageFont.load_default()

def rounded(draw, box, fill, outline=None, r=20):
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=2 if outline else 1)

def center_text(draw, box, text, f, fill, spacing=4):
    l,t,r,b=box; lines=text.split("\n")
    heights=[]
    for ln in lines:
        bb=draw.textbbox((0,0), ln, font=f); heights.append(bb[3]-bb[1])
    y=(t+b-sum(heights)-spacing*(len(lines)-1))/2
    for ln,h in zip(lines,heights):
        bb=draw.textbbox((0,0),ln,font=f); w=bb[2]-bb[0]
        draw.text(((l+r-w)/2,y),ln,font=f,fill=fill); y+=h+spacing

def arrow(draw, x1, y1, x2, y2, color='#2E74B5'):
    draw.line((x1,y1,x2,y2), fill=color, width=5)
    import math
    ang=math.atan2(y2-y1,x2-x1); s=16
    p1=(x2-s*math.cos(ang-.55),y2-s*math.sin(ang-.55)); p2=(x2-s*math.cos(ang+.55),y2-s*math.sin(ang+.55))
    draw.polygon([(x2,y2),p1,p2],fill=color)

def make_architecture():
    im=Image.new("RGB",(1500,780),"white"); d=ImageDraw.Draw(im)
    title=get_font(42,True); h=get_font(28,True); b=get_font(22)
    d.text((60,35),"项目总体技术路线",font=title,fill="#17365D")
    cols=[(60,200,350,470,"传感器与数据\n输入","单路 LiDAR\n六路鱼眼图像\n仿真 RGB/语义标签", "#E8F1FB"),
          (455,200,785,470,"分路感知","点云找目标与跟踪\n图像目标检测器训练、评估", "#E9F7F5"),
          (890,200,1215,470,"时空关联","图像目标与 LiDAR 目标\n多帧对应", "#FFF4E5"),
          (1320,200,1460,470,"输出","BEV 俯视图\n稳定目标轨迹", "#F0ECFA")]
    for l,t,r,bb,head,body,fill in cols:
        rounded(d,(l,t,r,bb),fill,"#CBD5E1",22)
        d.rectangle((l,t,r,t+66),fill="#17365D")
        center_text(d,(l,t,r,t+66),head,h,"white")
        center_text(d,(l+16,t+78,r-16,bb-14),body,b,"#1F2937",8)
    for a,bx in [(350,455),(785,890),(1215,1320)]: arrow(d,a,335,bx,335)
    d.text((62,575),"核心思想：先把每一路传感器做稳，再以时间连续性和空间位置为约束完成跨模态融合。",font=get_font(28),fill="#374151")
    d.text((62,635),"说明：图像检测器的具体网络型号尚未在现有材料中明确；本图呈现已确认的系统分工与后续接口。",font=get_font(22),fill="#5B6470")
    p=ASSETS/"architecture.png"; im.save(p); return p

def make_roadmap():
    im=Image.new("RGB",(1500,720),"white"); d=ImageDraw.Draw(im)
    d.text((60,35),"工作推进与验证闭环",font=get_font(42,True),fill="#17365D")
    ys=230; xs=[160,450,740,1030,1320]
    labels=[("1","样本与仿真","仿真生成 RGB 与\n语义标签；样本收集流程已攻克"),
            ("2","单 LiDAR 感知","点云目标发现、\n连续跟踪、BEV 输出"),
            ("3","ROS 2 联调","感知链路\n已完成验证"),
            ("4","六路图像检测","三 bag 训练与\n独立留出评估通过"),
            ("5","多模态融合","检测稳定后进行\n多帧图像-LiDAR 对应")]
    for i,(num,head,body) in enumerate(labels):
        color="#0F766E" if i<4 else "#64748B"
        d.ellipse((xs[i]-55,ys-55,xs[i]+55,ys+55),fill=color)
        center_text(d,(xs[i]-55,ys-55,xs[i]+55,ys+55),num,get_font(42,True),"white")
        center_text(d,(xs[i]-120,ys+85,xs[i]+120,ys+135),head,get_font(25,True),"#17365D")
        center_text(d,(xs[i]-130,ys+150,xs[i]+130,ys+245),body,get_font(21),"#374151",6)
        if i<4: arrow(d,xs[i]+62,ys,xs[i+1]-62,ys,"#94A3B8")
    d.text((65,590),"状态图例：绿色=已完成/已验证；灰色=下一阶段。",font=get_font(24),fill="#5B6470")
    p=ASSETS/"roadmap.png"; im.save(p); return p

def set_cell_shading(cell, fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),fill); tcPr.append(shd)

def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc=cell._tc; tcPr=tc.get_or_add_tcPr(); mar=tcPr.first_child_found_in('w:tcMar')
    if mar is None: mar=OxmlElement('w:tcMar'); tcPr.append(mar)
    for m,v in [('top',top),('start',start),('bottom',bottom),('end',end)]:
        node=mar.find(qn('w:'+m))
        if node is None: node=OxmlElement('w:'+m); mar.append(node)
        node.set(qn('w:w'),str(v)); node.set(qn('w:type'),'dxa')

def set_run(run, size=11, bold=False, color=INK):
    run.font.name='Microsoft YaHei'; run._element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei'); run._element.rPr.rFonts.set(qn('w:ascii'),'Calibri'); run.font.size=Pt(size); run.bold=bold; run.font.color.rgb=RGBColor.from_string(color)

def add_para(doc, text='', style=None, size=11, bold=False, color=INK, after=6, before=0, align=None):
    p=doc.add_paragraph(style=style); p.paragraph_format.space_before=Pt(before); p.paragraph_format.space_after=Pt(after); p.paragraph_format.line_spacing=1.1
    if align is not None: p.alignment=align
    r=p.add_run(text); set_run(r,size,bold,color); return p

def add_bullets(doc, lines):
    for x in lines:
        p=doc.add_paragraph(style='List Bullet'); p.paragraph_format.space_after=Pt(5); p.paragraph_format.line_spacing=1.1
        set_run(p.add_run(x),11,False,INK)

def add_caption(doc, text):
    p=add_para(doc,text,size=9.5,color=MUTED,after=12,align=WD_ALIGN_PARAGRAPH.CENTER); return p

def set_table_width(table, widths):
    table.alignment=WD_TABLE_ALIGNMENT.LEFT; table.autofit=False
    for row in table.rows:
        for cell,w in zip(row.cells,widths):
            cell.width=Inches(w); set_cell_margins(cell); cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER

def add_status_table(doc):
    table=doc.add_table(rows=1, cols=3); table.style='Table Grid'; set_table_width(table,[1.45,3.75,1.30])
    headers=['工作模块','已形成的能力/当前任务','状态']
    for cell,txt in zip(table.rows[0].cells,headers):
        set_cell_shading(cell,BLUE); p=cell.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER; set_run(p.add_run(txt),10.5,True,'FFFFFF')
    rows=[('LiDAR 感知','单路点云目标发现、连续跟踪，并生成 BEV 俯视结果。','已完成'),('系统链路','ROS 2 感知链路已验证。','已验证'),('仿真数据','可生成 RGB 图与语义标签，用于离线构建训练数据。','已具备'),('六路图像','YOLO11n 在两条人工标注 bag 训练后，已通过独立第三 bag 的六路正式留出评估。','已验证'),('跨模态融合','以已验证的静态图像候选和稳定 LiDAR 簇开展多帧候选优化。','下一阶段')]
    for i,row in enumerate(rows):
        cells=table.add_row().cells
        for j,(cell,txt) in enumerate(zip(cells,row)):
            if i%2==0: set_cell_shading(cell,'F7FAFC')
            p=cell.paragraphs[0]; p.paragraph_format.space_after=Pt(0); p.alignment=WD_ALIGN_PARAGRAPH.CENTER if j==2 else WD_ALIGN_PARAGRAPH.LEFT
            set_run(p.add_run(txt),10.3,j==0,INK)
    doc.add_paragraph().paragraph_format.space_after=Pt(4)

def main():
    arch=make_architecture(); roadmap=make_roadmap(); doc=Document(); sec=doc.sections[0]
    sec.top_margin=Inches(0.75); sec.bottom_margin=Inches(0.72); sec.left_margin=Inches(0.78); sec.right_margin=Inches(0.78); sec.header_distance=Inches(.35); sec.footer_distance=Inches(.35)
    styles=doc.styles
    normal=styles['Normal']; normal.font.name='Microsoft YaHei'; normal._element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei'); normal.font.size=Pt(11); normal.paragraph_format.space_after=Pt(6); normal.paragraph_format.line_spacing=1.1
    for name,size,col,bef,aft in [('Heading 1',16,BLUE,16,8),('Heading 2',13,BLUE,12,6),('Heading 3',11.5,NAVY,8,4)]:
        st=styles[name]; st.font.name='Microsoft YaHei'; st._element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei'); st.font.size=Pt(size); st.font.bold=True; st.font.color.rgb=RGBColor.from_string(col); st.paragraph_format.space_before=Pt(bef); st.paragraph_format.space_after=Pt(aft); st.paragraph_format.line_spacing=1.1
    # header/footer
    hp=sec.header.paragraphs[0]; hp.alignment=WD_ALIGN_PARAGRAPH.RIGHT; set_run(hp.add_run('多传感器环境感知项目 | 工作情况介绍'),9,False,MUTED)
    fp=sec.footer.paragraphs[0]; fp.alignment=WD_ALIGN_PARAGRAPH.CENTER; set_run(fp.add_run('基于《感知工作情况说明》整理 | 2026 年 8 月'),9,False,MUTED)
    # Cover
    add_para(doc,'项目工作情况介绍',size=13,bold=True,color=TEAL,after=18,align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc,'多传感器环境感知系统',size=28,bold=True,color=NAVY,after=8,align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc,'从单传感器稳定感知，到六路视觉与 LiDAR 的时空融合',size=14,color=MUTED,after=32,align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_picture(str(arch),width=Inches(6.8)); add_caption(doc,'图 1  系统总体技术路线（依据现有工作说明整理）')
    add_para(doc,'内容说明',size=13,bold=True,color=BLUE,after=6)
    add_para(doc,'本介绍以当前已完成、已验证及正在推进的工作为边界展开。图像侧已冻结 YOLO11n 原始鱼眼候选模型：仅用两条人工标注 bag 训练，并在独立第三条 bag 的全部 240 张标注图上按预先声明的门槛完成一次性评估；六路均通过。该结果允许进入多帧候选优化，但不等同于外参验收或像素级融合许可。',size=10.5,color=INK,after=10)
    doc.add_page_break()
    # Content
    add_para(doc,'一、项目目标与起始设计思路',style='Heading 1')
    add_para(doc,'项目面向复杂环境中的目标感知问题，目标不是仅在单帧中“看见”目标，而是形成可连续运行、可验证、可扩展的系统能力：一方面从 LiDAR 点云中发现目标并保持轨迹连续；另一方面利用六路鱼眼相机补充目标的外观与语义信息；最终把图像与点云中的同一目标在时间与空间上对应起来，并以 BEV（鸟瞰图）形式输出更直观的环境理解结果。')
    add_para(doc,'为什么采用“LiDAR + 多相机 + ROS 2”的组合？',style='Heading 2')
    add_bullets(doc,[
        'LiDAR 直接提供三维几何与距离信息，适合承担目标定位、空间结构表达以及连续跟踪的基础职责。',
        '鱼眼相机具有更宽的覆盖范围，六路布置可降低单一视角的遮挡和盲区风险，并为类别、外观等语义理解提供信息来源。',
        'ROS 2 用于组织传感器数据、感知节点与结果输出，先完成链路验证可降低后续模型接入和多模块协同的工程不确定性。',
        '仿真环境可同步产生 RGB 图和语义标签，使真实采集、人工标注与离线训练之间形成可重复的数据补给路径。'
    ])
    add_para(doc,'设计策略上，项目没有一开始就把所有功能堆叠到一起，而是采用“分路做稳、链路打通、再做融合”的递进路径。这样做的原因是多传感器融合的误差会来自标定、时间同步、检测漏检、跟踪漂移等多个环节；先验证每一路的可用性，才能更清楚地定位后续融合问题的来源。',after=10)
    add_para(doc,'二、为什么需要机器学习，以及模型在系统中的位置',style='Heading 1')
    add_para(doc,'在图像侧，目标的外观、尺度、姿态、光照和遮挡会持续变化，尤其鱼眼成像还存在明显的几何畸变。单纯依靠人工规则难以覆盖这些变化，因此项目采用以机器学习目标检测器为核心的图像感知路线：通过已标注的鱼眼图像学习“目标外观—位置”的映射，在新的图像中输出候选目标框、类别与置信度（具体输出以所选检测器的定义为准）。')
    add_para(doc,'“用了什么模型”的事实边界',style='Heading 2')
    add_para(doc,'图像侧冻结候选模型为 YOLO11n，在原始 1280×720 鱼眼图域以 960 px 运行。训练数据仅来自两条人工标注 bag；独立第三条 bag 不参与训练、阈值选择或模型选择。以训练前固定的 confidence=0.50、IoU>=0.20 评估时，六路均满足 precision>=0.90、matches>=20、median center error<=12 px 的门槛。LiDAR 侧已经具备点云找目标与连续跟踪能力；融合侧仍只计划利用多帧图像目标与 LiDAR 目标之间的时间连续性与空间一致性建立候选对应。')
    add_para(doc,'模型选择的工程准则',style='Heading 2')
    add_bullets(doc,[
        '准确性：在各个鱼眼视角、不同距离与遮挡条件下保持稳定检出。',
        '鲁棒性：面对鱼眼畸变、亮度变化和目标尺度变化时，误检与漏检可控。',
        '实时性：模型推理速度需要与 ROS 2 数据流及多帧关联节奏相匹配。',
        '可维护性：训练、评估、数据版本与部署链路能够复现，便于后续迭代。'
    ])
    doc.add_picture(str(roadmap),width=Inches(6.8)); add_caption(doc,'图 2  当前工作推进节奏与后续闭环')
    add_para(doc,'三、已完成与正在推进的工作',style='Heading 1')
    add_status_table(doc)
    add_para(doc,'已完成的关键能力：LiDAR 单路闭环',style='Heading 2')
    add_para(doc,'当前已完成从单路 LiDAR 点云中寻找目标并连续跟踪的能力，最终可生成 BEV 俯视图。这意味着系统已经不只是输出某一时刻的孤立检测结果，而是能够把连续观测组织为时序轨迹；BEV 输出则把三维空间理解转化为更适合观察目标位置关系与运动趋势的统一视角。')
    add_para(doc,'已验证的系统工程基础：ROS 2 链路',style='Heading 2')
    add_para(doc,'ROS 2 链路已经验证，说明感知模块之间的数据传递、节点组织或结果发布已具备可用基础。后续把图像检测器、LiDAR 跟踪器与关联模块接入时，可在同一通信框架下逐步验证输入、输出、时间戳和异常处理，而不必等到全部算法完成后再进行系统集成。')
    add_para(doc,'已验证的数据与训练能力：六路鱼眼图像',style='Heading 2')
    add_para(doc,'团队已完成六路原始鱼眼图的人工标注、训练与独立评估。不同相机视角的畸变分布、目标尺寸和背景可能不同，因此按相机分别验收，避免用整体平均结果掩盖某些视角的短板。冻结模型的 `buoy`、`pillar`、`block` 检测会携带时间戳、相机名、模型版本和置信度，以候选形式进入后续关联门禁；动态 `boat` 不作为静态标定对应。')
    add_para(doc,'四、数据闭环：仿真、标注与训练如何协同',style='Heading 1')
    add_para(doc,'仿真环境已经能够生成 RGB 图像与语义标签，可用于离线构建训练数据。这为项目提供了一个重要的数据杠杆：在真实采集受时间、覆盖场景或标注成本限制时，仿真数据可以补充具有明确标签的样本，支撑检测器的预训练、数据增强和极端场景覆盖。')
    add_bullets(doc,[
        '仿真生成：输出 RGB 图与语义标签，形成可直接用于训练或转换为检测标注的数据来源。',
        '真实数据标注：围绕六路原始鱼眼图建立可信训练、验证样本，保证模型面对真实成像特性时可用。',
        '分别训练与评估：按相机视角检查检测器表现，识别数据不足、畸变影响或场景差异。',
        '反馈迭代：将评估中发现的问题回流到样本补充、标注规范和训练配置，逐渐积累可用于跨模态匹配的稳定检测结果。'
    ])
    add_para(doc,'现有材料明确指出，较为繁琐的样本收集过程已经攻克，后续暂无阻碍。这说明项目当前的重心已从“能否获得可用数据”转向“提高标注与模型质量、稳定六路推理并验证融合效果”。',after=10)
    add_para(doc,'五、下一阶段：从分路感知走向多帧融合',style='Heading 1')
    add_para(doc,'下一阶段的核心不是简单地把两个检测结果放在一起，而是建立“图像目标—LiDAR 目标”的多帧对应。单帧关联容易受到遮挡、视角差异、误检与时间偏移影响；多帧策略则可以利用轨迹连续性、相对位置变化和重复观测来提升对应的可信度。')
    add_bullets(doc,[
        '先确保六路检测器在各自视角上运行稳定，并保留可追溯的评估结果。',
        '以时间戳、外参标定与投影关系为基础，给图像目标和 LiDAR 目标建立候选匹配。',
        '在连续多帧中利用轨迹一致性筛除偶然匹配，输出更可靠的跨模态对应。',
        '将融合结果回写到 BEV 或统一目标列表中，形成兼具几何位置与图像语义的环境表示。'
    ])
    add_para(doc,'六、当前实现效果与阶段性判断',style='Heading 1')
    add_para(doc,'从现有工作情况看，项目已经建立了“点云感知与跟踪—ROS 2 工程链路—仿真数据生产—六路图像训练评估—跨模态多帧候选对应”的完整技术闭环框架。其中，LiDAR 单路感知、连续跟踪、BEV 输出和 ROS 2 链路验证已经形成可展示的阶段性成果；图像侧已完成三 bag、六路独立验收；融合则具有清晰的后续接口和验证路径。')
    add_para(doc,'需要特别区分的是：检测器的离线留出验收已经通过，但外参、融合正确率和端到端实时性尚未完成最终验收。因此当前最准确的表述是“核心基础能力和六路视觉候选检测已验证，融合进入受门禁约束的候选优化阶段”，而非宣称全部系统已经完成最终性能验收。后续建议用每路检测精度、漏检/误检、端到端时延、轨迹连续性、跨模态匹配正确率等指标构成统一的验证看板。')
    add_para(doc,'结语',style='Heading 1')
    add_para(doc,'项目的价值在于把机器学习的图像语义能力、LiDAR 的三维几何能力和 ROS 2 的系统工程能力组合为一套可持续演进的感知系统。当前已经跨过了样本收集与基础链路验证这两个关键门槛；接下来只要持续完善六路标注与模型评估，并以多帧一致性推进图像—LiDAR 对应，就能把已有的单路成果扩展为面向全视角、多模态的稳定环境感知能力。')
    doc.save(OUT)
    print(OUT)

if __name__=='__main__': main()
