#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <Luigi Carmona de Miranda Lopes>
Disciplina: Computação Gráfica
Data: <12/08/2026>
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy
from itertools import batched, pairwise


class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante

    view_matrix = np.identity(4)         # matriz da câmera (mundo -> câmera)
    perspective_matrix = np.identity(4)  # matriz de projeção perspectiva
    transform_stack = [np.identity(4)]   # pilha de matrizes de transformação (mundo)

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far

    @staticmethod
    def rotation_matrix(axis, angle):
        """Cria uma matriz de rotação 4x4 (homogênea) a partir de eixo/ângulo."""
        x, y, z = axis
        norm = math.sqrt(x * x + y * y + z * z)
        if norm == 0:
            return np.identity(4)
        x, y, z = x / norm, y / norm, z / norm
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1 - c
        return np.array([
            [t*x*x + c,   t*x*y - s*z, t*x*z + s*y, 0],
            [t*x*y + s*z, t*y*y + c,   t*y*z - s*x, 0],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c,   0],
            [0,           0,           0,           1],
        ])

    @staticmethod
    def translation_matrix(translation):
        """Cria uma matriz de translação 4x4 (homogênea)."""
        matrix = np.identity(4)
        matrix[:3, 3] = translation
        return matrix

    @staticmethod
    def scale_matrix(scale):
        """Cria uma matriz de escala 4x4 (homogênea)."""
        matrix = np.identity(4)
        matrix[0, 0], matrix[1, 1], matrix[2, 2] = scale
        return matrix


    @staticmethod
    def draw2D(point, color):
        x, y = int(point[0]), int(point[1])
        if not (0 <= x < GL.width and 0 <= y < GL.height):
            return
        rgb = [int(i * 255) for i in color["emissiveColor"]]
        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, rgb)

    def bresenham(p0, p1):
        x0, y0 = round(p0[0]), round(p0[1])
        x1, y1 = round(p1[0]), round(p1[1])
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            yield x0, y0
            if x0 == x1 and y0 == y1:
                return
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    
    @staticmethod
    def inside(a, b, c, x, y):
        def edge(p, q):
            return (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0])

        w = (edge(a, b), edge(b, c), edge(c, a))
        return all(v >= 0 for v in w) or all(v <= 0 for v in w)

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).

        for x, y in batched(point, 2):
            GL.draw2D((x, y), colors)
        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polyline2D
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        for p0, p1 in pairwise(batched(lineSegments, 2)):
            for pixel in GL.bresenham(p0, p1):
                GL.draw2D(pixel, colors)        

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        print("Circle2D : radius = {0}".format(radius)) # imprime no terminal
        print("Circle2D : colors = {0}".format(colors)) # imprime no terminal as cores
        
        # Exemplo:
        pos_x = GL.width//2
        pos_y = GL.height//2
        gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 255])  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)


    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#TriangleSet2D
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).
        
        for tri in batched(vertices, 6):
            a, b, c = batched(tri, 2)
            for y in range(GL.height):
                for x in range(GL.width):
                    if GL.inside(a, b, c, x + 0.5, y + 0.5):
                        GL.draw2D((x, y), colors)


    @staticmethod
    def mvp_matrix():
        """Matriz completa que leva pontos do espaço do objeto para o espaço de tela:
        modelo (pilha de transform) -> câmera (view) -> projeção."""
        return GL.perspective_matrix @ GL.view_matrix @ GL.transform_stack[-1]

    @staticmethod
    def project(vertex, mvp):
        """Projeta um ponto 3D do espaço do objeto para coordenadas de tela (x, y)."""
        x, y, z = vertex
        clip = mvp @ np.array([x, y, z, 1.0])
        ndc = clip[:3] / clip[3]
        screen_x = (ndc[0] + 1) / 2 * GL.width
        screen_y = (1 - ndc[1]) / 2 * GL.height
        return screen_x, screen_y

    @staticmethod
    def fill_triangle(a, b, c, colors, vertex_colors=None, uvs=None, texture=None):
        """Percorre a bounding box de um triângulo em tela e pinta os pixels internos.

        Por padrão pinta tudo com colors["emissiveColor"] (cor plana). Se vertex_colors
        (3 cores RGB, uma por vértice) ou uvs (3 pares uv) + texture forem passados, a cor
        de cada pixel é interpolada por coordenadas baricêntricas.
        """
        min_x = max(int(min(a[0], b[0], c[0])), 0)
        max_x = min(int(max(a[0], b[0], c[0])) + 1, GL.width)
        min_y = max(int(min(a[1], b[1], c[1])), 0)
        max_y = min(int(max(a[1], b[1], c[1])) + 1, GL.height)

        def edge(p, q, x, y):
            return (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0])

        area = edge(a, b, c[0], c[1])
        if area == 0:
            return

        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                px, py = x + 0.5, y + 0.5
                if not GL.inside(a, b, c, px, py):
                    continue
                if vertex_colors is None and uvs is None:
                    GL.draw2D((x, y), colors)
                    continue

                # Pesos baricêntricos reaproveitando as mesmas funções de aresta do inside()
                wa = edge(b, c, px, py) / area
                wb = edge(c, a, px, py) / area
                wc = 1 - wa - wb

                if vertex_colors is not None:
                    rgb = [wa * vertex_colors[0][i] + wb * vertex_colors[1][i] + wc * vertex_colors[2][i]
                           for i in range(3)]
                else:
                    u = wa * uvs[0][0] + wb * uvs[1][0] + wc * uvs[2][0]
                    v = wa * uvs[0][1] + wb * uvs[1][1] + wc * uvs[2][1]
                    th, tw = texture.shape[0], texture.shape[1]
                    # ponytail: amostragem "nearest" sem filtragem; nenhum exemplo desta
                    # etapa usa textura, então a convenção de eixo u/v não foi validada
                    # visualmente. Ajustar se um exemplo com ImageTexture falhar.
                    tx = min(max(int(u * tw), 0), tw - 1)
                    ty = min(max(int((1 - v) * th), 0), th - 1)
                    rgb = [channel / 255 for channel in texture[ty, tx][:3]]

                GL.draw2D((x, y), {**colors, "emissiveColor": rgb})

    @staticmethod
    def split_strips(indices):
        """Separa uma lista de índices em tiras, quebrando a cada -1."""
        strips = []
        current = []
        for i in indices:
            if i == -1:
                if current:
                    strips.append(current)
                current = []
            else:
                current.append(i)
        if current:
            strips.append(current)
        return strips

    @staticmethod
    def draw_strip(screen_points, colors, vertex_colors=None):
        """Desenha uma tira de triângulos já projetada em tela (índices 0, 1, 2, depois
        1, 2, 3, etc.), alternando a ordem dos vértices para manter a orientação."""
        for i in range(len(screen_points) - 2):
            tri = (0, 1, 2) if i % 2 == 0 else (1, 0, 2)
            a, b, c = (screen_points[i + tri[0]], screen_points[i + tri[1]], screen_points[i + tri[2]])
            vc = None
            if vertex_colors is not None:
                vc = (vertex_colors[i + tri[0]], vertex_colors[i + tri[1]], vertex_colors[i + tri[2]])
            GL.fill_triangle(a, b, c, colors, vertex_colors=vc)

    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.

        mvp = GL.mvp_matrix()
        for tri in batched(point, 9):
            a, b, c = (GL.project(p, mvp) for p in batched(tri, 3))
            GL.fill_triangle(a, b, c, colors)

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        # Matriz da câmera no mundo (posição e orientação) e sua inversa (view matrix),
        # que leva pontos do sistema de coordenadas do mundo para o da câmera
        rotation = GL.rotation_matrix(orientation[:3], orientation[3])
        translation = GL.translation_matrix(position)
        camera_matrix = translation @ rotation
        GL.view_matrix = np.linalg.inv(camera_matrix)

        # O fieldOfView se aplica à menor dimensão da tela, a outra é derivada pela
        # razão de aspecto para não distorcer a imagem.
        aspect_ratio = GL.width / GL.height
        fovy = fieldOfView
        if aspect_ratio < 1:
            fovy = 2 * math.atan(math.tan(fieldOfView / 2) / aspect_ratio)

        top = GL.near * math.tan(fovy / 2)
        right = top * aspect_ratio
        near, far = GL.near, GL.far

        GL.perspective_matrix = np.array([
            [near / right, 0, 0, 0],
            [0, near / top, 0, 0],
            [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, -1, 0],
        ])

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas.
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.

        # A ordem de aplicação em um ponto local é escala, depois rotação, depois translação
        t_matrix = GL.translation_matrix(translation)
        r_matrix = GL.rotation_matrix(rotation[:3], rotation[3])
        s_matrix = GL.scale_matrix(scale)
        local_matrix = t_matrix @ r_matrix @ s_matrix

        parent_matrix = GL.transform_stack[-1]
        GL.transform_stack.append(parent_matrix @ local_matrix)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.
        GL.transform_stack.pop()

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # No TriangleStripSet a quantidade de vértices de cada tira é informada em stripCount
        # (uma lista, pois pode haver várias tiras). Ligue os vértices na ordem, primeiro
        # triângulo com os vértices 0, 1 e 2, depois 1, 2 e 3, depois 2, 3 e 4, e assim por
        # diante, alternando a orientação para manter a face sempre no mesmo sentido.

        mvp = GL.mvp_matrix()
        points = list(batched(point, 3))
        offset = 0
        for count in stripCount:
            strip = points[offset:offset + count]
            offset += count
            screen = [GL.project(p, mvp) for p in strip]
            GL.draw_strip(screen, colors)

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # No IndexedTriangleStripSet uma lista informando como conectar os vértices é
        # informada em index; o valor -1 separa uma tira da outra. Dentro de cada tira, os
        # triângulos são formados 0-1-2, 1-2-3, 2-3-4, etc., alternando a orientação.

        mvp = GL.mvp_matrix()
        points = list(batched(point, 3))
        for strip in GL.split_strips(index):
            screen = [GL.project(points[i], mvp) for i in strip]
            GL.draw_strip(screen, colors)

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # coordIndex lista os vértices de cada face, separadas por -1. Cada face é
        # triangulada em leque a partir do primeiro vértice: 0-1-2, 0-2-3, 0-3-4, etc.
        # Se colorPerVertex e color/colorIndex forem dados, a cor é interpolada por
        # baricêntricas; senão cada face usa uma única cor (colorIndex) ou a cor do material.
        # Se texCoord/texCoordIndex e uma textura forem dados, a cor vem da textura.

        if not coord:
            return

        mvp = GL.mvp_matrix()
        points = list(batched(coord, 3))
        faces = GL.split_strips(coordIndex)

        colors_list = list(batched(color, 3)) if color else None
        color_faces = GL.split_strips(colorIndex) if colorIndex else faces

        uv_list = list(batched(texCoord, 2)) if texCoord else None
        uv_faces = GL.split_strips(texCoordIndex) if texCoordIndex else faces

        texture = None
        if current_texture and uv_list:
            texture = gpu.GPU.load_texture(current_texture[0])

        for face_i, face in enumerate(faces):
            screen = [GL.project(points[i], mvp) for i in face]

            face_colors = colors
            vertex_colors = None
            if texture is not None and face_i < len(uv_faces):
                uvs_face = [uv_list[i] for i in uv_faces[face_i]]
            else:
                uvs_face = None

            if colorPerVertex and colors_list and face_i < len(color_faces):
                vertex_colors = [colors_list[i] for i in color_faces[face_i]]
            elif colors_list and face_i < len(color_faces) and color_faces[face_i]:
                flat_rgb = colors_list[color_faces[face_i][0]]
                face_colors = {**colors, "emissiveColor": flat_rgb}

            for i in range(1, len(screen) - 1):
                tri = (0, i, i + 1)
                a, b, c = screen[tri[0]], screen[tri[1]], screen[tri[2]]
                vc = [vertex_colors[t] for t in tri] if vertex_colors else None
                uvs = [uvs_face[t] for t in tri] if uvs_face else None
                GL.fill_triangle(a, b, c, face_colors, vertex_colors=vc, uvs=uvs,
                                  texture=texture if uvs else None)

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cone : bottomRadius = {0}".format(bottomRadius)) # imprime no terminal o raio da base do cone
        print("Cone : height = {0}".format(height)) # imprime no terminal a altura do cone
        print("Cone : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cylinder : radius = {0}".format(radius)) # imprime no terminal o raio do cilindro
        print("Cylinder : height = {0}".format(height)) # imprime no terminal a altura do cilindro
        print("Cylinder : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
