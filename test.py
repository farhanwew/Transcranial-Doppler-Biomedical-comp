import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):

    def __init__(self):
        # Inisialisasi node dengan nama 'minimal_subscriber'
        super().__init__('minimal_subscriber')
        # Membuat subscription untuk mendengarkan topik 'topic' dengan pesan tipe String
        # Fungsi listener_callback akan dipanggil setiap kali pesan diterima
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10) # Parameter QoS depth
        self.subscription  # Mencegah peringatan variabel tidak terpakai

    def listener_callback(self, msg):
        # Fungsi ini dieksekusi saat pesan baru diterima
        self.get_logger().info(f'Saya mendengar: "{msg.data}"') # Menampilkan data pesan

def main(args=None):
    # Fungsi utama untuk menginisialisasi dan menjalankan node
    rclpy.init(args=args) # Inisialisasi ROS2
    minimal_subscriber = MinimalSubscriber() # Membuat instance node subscriber
    rclpy.spin(minimal_subscriber) # Menjaga node tetap berjalan
    # Membersihkan sumber daya saat node dihentikan
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
