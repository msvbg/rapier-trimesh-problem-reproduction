use lazy_static::lazy_static;
use lyon::{
    geom::euclid::Point2D, lyon_tessellation::BuffersBuilder, lyon_tessellation::FillAttributes,
    lyon_tessellation::FillOptions, lyon_tessellation::FillTessellator,
    lyon_tessellation::FillVertexConstructor, lyon_tessellation::VertexBuffers,
};
use nalgebra::{Point3, Vector2, Vector3};
use rapier2d::{
    geometry::{Collider, ColliderBuilder, Shape, Trimesh},
    math::Point,
};

#[derive(Clone)]
pub struct Polygon {
    pub vertices: Vec<Vector3<f32>>,
}

pub struct Level {
    pub geometry: Vec<Polygon>,
}

lazy_static! {
    pub static ref TEST_LEVEL: Level = Level {
        geometry: vec![Polygon {
            vertices: vec![
                Vector3::<f32>::new(1.0, 0.0, 0.0),
                Vector3::<f32>::new(1.0, 1.0, 0.0),
                Vector3::<f32>::new(0.0, 1.0, 0.0),
                Vector3::<f32>::new(0.6, 0.6, 0.0),
            ]
        }]
    };
}

fn tessellate_geometry<Vertex, VertexConstructor: FillVertexConstructor<Vertex> + Default>(
    geometry: &Vec<Polygon>,
) -> VertexBuffers<Vertex, u16> {
    use lyon::path::Path;
    let mut path_builder = Path::builder();

    for polygon in geometry {
        let mut first_point = true;
        for v in &polygon.vertices {
            if first_point {
                path_builder.move_to(Point2D::new(v.x, v.y));
                first_point = false;
            } else {
                path_builder.line_to(Point2D::new(v.x, v.y));
            }
        }
    }
    path_builder.close();

    let path = path_builder.build();
    let mut buffers: VertexBuffers<Vertex, u16> = VertexBuffers::new();

    let mut vertex_builder = BuffersBuilder::new(&mut buffers, VertexConstructor::default());
    let mut tessellator = FillTessellator::new();
    let result = tessellator.tessellate_path(&path, &FillOptions::default(), &mut vertex_builder);

    if result.is_ok() {
        buffers
    } else {
        panic!("Failed to tessellate level");
    }
}

impl Level {
    pub fn tessellate<Vertex, VertexConstructor: FillVertexConstructor<Vertex> + Default>(
        &self,
    ) -> VertexBuffers<Vertex, u16> {
        tessellate_geometry::<Vertex, VertexConstructor>(&self.geometry)
    }

    pub fn get_collider(&self) -> Collider {
        let vertex_buffers = self.tessellate::<Point<f32>, VertexConstructor>();
        let indices = vertex_buffers
            .indices
            .chunks(3)
            .map(|chunk| match chunk {
                [x, y, z] => Point3::new(*x as u32, *y as u32, *z as u32),
                _ => panic!("Index buffer length not divisible by three"),
            })
            .collect::<Vec<Point3<u32>>>();
        ColliderBuilder::new(Shape::Trimesh(Trimesh::new(
            vertex_buffers.vertices,
            indices,
        )))
        .build()
    }
}

#[derive(Default)]
struct VertexConstructor;

impl FillVertexConstructor<Point<f32>> for VertexConstructor {
    fn new_vertex(
        &mut self,
        position: lyon::geom::math::Point,
        _attributes: FillAttributes,
    ) -> Point<f32> {
        Point {
            coords: Vector2::<f32>::new(position.x, position.y),
        }
    }
}
