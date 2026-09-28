#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Renderizador X3D.

Desenvolvido por: Luciano Soares <lpsoares@insper.edu.br>
Disciplina: Computação Gráfica
Data: 28 de Agosto de 2020
"""

import os           # Para rotinas do sistema operacional
import argparse     # Para tratar os parâmetros da linha de comando

import gl           # Recupera rotinas de suporte ao X3D

import interface    # Janela de visualização baseada no Matplotlib
import gpu          # Simula os recursos de uma GPU

import x3d          # Faz a leitura do arquivo X3D, gera o grafo de cena e faz traversal
import scenegraph   # Imprime o grafo de cena no console

LARGURA = 60  # Valor padrão para largura da tela
ALTURA = 40   # Valor padrão para altura da tela


class Renderizador:
    """Realiza a renderização da cena informada."""

    def __init__(self):
        """Definindo valores padrão."""
        self.width = LARGURA
        self.height = ALTURA
        self.ssaa = 2  # fator de supersampling (anti-aliasing), ligado por padrão
        self.x3d_file = ""
        self.image_file = "tela.png"
        self.scene = None
        self.framebuffers = {}

    def setup(self):
        """Configura o sistema para a renderização."""
        # Configurando color buffers para exibição na tela

        # Cria duas posições de FrameBuffer na GPU:
        # - SS: onde a GL de fato desenha, numa resolução "ssaa" vezes maior (para o
        #   anti-aliasing por supersampling) e com canal de profundidade (z-buffer).
        # - FRONT: buffer na resolução final, só de cor, onde vai parar a imagem já
        #   com o downsampling feito (é o que é exibido/salvo).
        fbo = gpu.GPU.gen_framebuffers(2)
        self.framebuffers["SS"] = fbo[0]
        self.framebuffers["FRONT"] = fbo[1]

        largura_ss = self.width * self.ssaa
        altura_ss = self.height * self.ssaa

        gpu.GPU.framebuffer_storage(
            self.framebuffers["SS"],
            gpu.GPU.COLOR_ATTACHMENT,
            gpu.GPU.RGB8,
            largura_ss,
            altura_ss
        )
        gpu.GPU.framebuffer_storage(
            self.framebuffers["SS"],
            gpu.GPU.DEPTH_ATTACHMENT,
            gpu.GPU.DEPTH_COMPONENT32F,
            largura_ss,
            altura_ss
        )

        gpu.GPU.framebuffer_storage(
            self.framebuffers["FRONT"],
            gpu.GPU.COLOR_ATTACHMENT,
            gpu.GPU.RGB8,
            self.width,
            self.height
        )

        # Opções:
        # - COLOR_ATTACHMENT: alocações para as cores da imagem renderizada
        # - DEPTH_ATTACHMENT: alocações para as profundidades da imagem renderizada

        # Tipos de dados:
        # - RGB8: Para canais de cores (Vermelho, Verde, Azul) 8bits cada (0-255)
        # - RGBA8: Para canais de cores (Vermelho, Verde, Azul, Transparência) 8bits cada (0-255)
        # - DEPTH_COMPONENT16: Para canal de Profundidade de 16bits (half-precision) (0-65535)
        # - DEPTH_COMPONENT32F: Para canal de Profundidade de 32bits (single-precision) (float)

        # Define cor que ira apagar o FrameBuffer quando clear_buffer() invocado
        gpu.GPU.clear_color([0, 0, 0])

        # Define a profundidade que ira apagar o FrameBuffer quando clear_buffer() invocado
        # Assuma 1.0 o mais afastado e -1.0 o mais próximo da camera
        gpu.GPU.clear_depth(1.0)

        # Definindo tamanho do Viewport para renderização (já na resolução supersampleada)
        self.scene.viewport(width=largura_ss, height=altura_ss)

    def pre(self):
        """Rotinas pré renderização."""
        # Função invocada antes do processo de renderização iniciar.

        # A GL sempre desenha (e faz o teste de profundidade) no buffer supersampleado
        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, self.framebuffers["SS"])

        # Limpa o frame buffers atual
        gpu.GPU.clear_buffer()

        # Recursos que podem ser úteis:
        # Define o valor do pixel no framebuffer: draw_pixel(coord, mode, data)
        # Retorna o valor do pixel no framebuffer: read_pixel(coord, mode)

    def pos(self):
        """Rotinas pós renderização."""
        # Função invocada após o processo de renderização terminar.

        # Essa é uma chamada conveniente para manipulação de buffers
        # ao final da renderização de um frame. Como por exemplo, executar
        # downscaling da imagem.

        # Faz o downsampling: lê blocos ssaa x ssaa do buffer supersampleado (SS) e
        # escreve a média de cada bloco como um pixel no buffer final (FRONT).
        gpu.GPU.bind_framebuffer(gpu.GPU.READ_FRAMEBUFFER, self.framebuffers["SS"])
        gpu.GPU.bind_framebuffer(gpu.GPU.DRAW_FRAMEBUFFER, self.framebuffers["FRONT"])
        gpu.GPU.clear_buffer()  # limpa o FRONT antes de escrever a imagem nova

        amostras_por_pixel = self.ssaa * self.ssaa
        for y in range(self.height):
            for x in range(self.width):
                soma_r, soma_g, soma_b = 0, 0, 0
                for dy in range(self.ssaa):
                    for dx in range(self.ssaa):
                        r, g, b = gpu.GPU.read_pixel(
                            [x * self.ssaa + dx, y * self.ssaa + dy], gpu.GPU.RGB8)
        
                        # estoura (satura em 255) ao somar mais de um pixel não totalmente preto
                        soma_r += int(r)
                        soma_g += int(g)
                        soma_b += int(b)
                media = [soma_r // amostras_por_pixel,
                         soma_g // amostras_por_pixel,
                         soma_b // amostras_por_pixel]
                gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, media)

        # Deixa o FRONT (resolução final) como buffer de leitura para exibição/salvamento
        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, self.framebuffers["FRONT"])

        # Método para a troca dos buffers (NÃO IMPLEMENTADO)
        # Esse método será utilizado na fase de implementação de animações
        gpu.GPU.swap_buffers()

    def mapping(self):
        """Mapeamento de funções para as rotinas de renderização."""
        # Rotinas encapsuladas na classe GL (Graphics Library)
        x3d.X3D.renderer["Polypoint2D"] = gl.GL.polypoint2D
        x3d.X3D.renderer["Polyline2D"] = gl.GL.polyline2D
        x3d.X3D.renderer["Circle2D"] = gl.GL.circle2D
        x3d.X3D.renderer["TriangleSet2D"] = gl.GL.triangleSet2D
        x3d.X3D.renderer["TriangleSet"] = gl.GL.triangleSet
        x3d.X3D.renderer["Viewpoint"] = gl.GL.viewpoint
        x3d.X3D.renderer["Transform_in"] = gl.GL.transform_in
        x3d.X3D.renderer["Transform_out"] = gl.GL.transform_out
        x3d.X3D.renderer["TriangleStripSet"] = gl.GL.triangleStripSet
        x3d.X3D.renderer["IndexedTriangleStripSet"] = gl.GL.indexedTriangleStripSet
        x3d.X3D.renderer["IndexedFaceSet"] = gl.GL.indexedFaceSet
        x3d.X3D.renderer["Box"] = gl.GL.box
        x3d.X3D.renderer["Sphere"] = gl.GL.sphere
        x3d.X3D.renderer["Cone"] = gl.GL.cone
        x3d.X3D.renderer["Cylinder"] = gl.GL.cylinder
        x3d.X3D.renderer["NavigationInfo"] = gl.GL.navigationInfo
        x3d.X3D.renderer["DirectionalLight"] = gl.GL.directionalLight
        x3d.X3D.renderer["PointLight"] = gl.GL.pointLight
        x3d.X3D.renderer["Fog"] = gl.GL.fog
        x3d.X3D.renderer["TimeSensor"] = gl.GL.timeSensor
        x3d.X3D.renderer["SplinePositionInterpolator"] = gl.GL.splinePositionInterpolator
        x3d.X3D.renderer["OrientationInterpolator"] = gl.GL.orientationInterpolator

    def render(self):
        """Laço principal de renderização."""
        self.pre()  # executa rotina pré renderização
        self.scene.render()  # faz o traversal no grafo de cena
        self.pos()  # executa rotina pós renderização
        return gpu.GPU.get_frame_buffer()

    def main(self):
        """Executa a renderização."""
        # Tratando entrada de parâmetro
        parser = argparse.ArgumentParser(add_help=False)   # parser para linha de comando
        parser.add_argument("-i", "--input", help="arquivo X3D de entrada")
        parser.add_argument("-o", "--output", help="arquivo 2D de saída (imagem)")
        parser.add_argument("-w", "--width", help="resolução horizonta", type=int)
        parser.add_argument("-h", "--height", help="resolução vertical", type=int)
        parser.add_argument("-g", "--graph", help="imprime o grafo de cena", action='store_true')
        parser.add_argument("-p", "--pause", help="começa simulação em pausa", action='store_true')
        parser.add_argument("-q", "--quiet", help="não exibe janela", action='store_true')
        args = parser.parse_args() # parse the arguments
        if args.input:
            self.x3d_file = args.input
        if args.output:
            self.image_file = args.output
        if args.width:
            self.width = args.width
        if args.height:
            self.height = args.height

        path = os.path.dirname(os.path.abspath(self.x3d_file))

        # Iniciando simulação de GPU
        gpu.GPU(self.image_file, path)

        # Abre arquivo X3D
        self.scene = x3d.X3D(self.x3d_file)

        # Iniciando Biblioteca Gráfica
        gl.GL.setup(
            self.width,
            self.height,
            near=0.01,
            far=1000,
            ssaa=self.ssaa
        )

        # Funções que irão fazer o rendering
        self.mapping()

        # Se no modo silencioso não configurar janela de visualização
        if not args.quiet:
            window = interface.Interface(self.width, self.height, self.x3d_file)
            self.scene.set_preview(window)

        # carrega os dados do grafo de cena
        if self.scene:
            self.scene.parse()
            if args.graph:
                scenegraph.Graph(self.scene.root)

        # Configura o sistema para a renderização.
        self.setup()

        # Se no modo silencioso salvar imagem e não mostrar janela de visualização
        if args.quiet:
            gpu.GPU.save_image()  # Salva imagem em arquivo
        else:
            window.set_saver(gpu.GPU.save_image)  # pasa a função para salvar imagens
            window.preview(args.pause, self.render)  # mostra visualização

if __name__ == '__main__':
    renderizador = Renderizador()
    renderizador.main()
